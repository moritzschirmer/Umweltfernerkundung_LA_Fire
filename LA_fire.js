var Perimeters = ee.FeatureCollection('projects/ee-moritzschirmer02/assets/Perimeters');    //wird ersetzt durch import

Map.addLayer(Perimeters, {}, 'Perimeters');

// Los Angeles County
var losAngelesCounty = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('COUNTYFP', '037'));

// Landsat Collection
var landsatCollection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(losAngelesCounty)
  .filterDate('2024-12-01', '2025-12-31');


// Cloud Mask
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0);
  var shadowMask = qa.bitwiseAnd(1 << 4).eq(0);
  return image.updateMask(cloudMask).updateMask(shadowMask);
}


// Indizes berechnen
function addIndices(image){

  var ndvi = image.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
  var nbr  = image.normalizedDifference(['SR_B5','SR_B7']).rename('NBR');
  var ndbi = image.normalizedDifference(['SR_B6','SR_B5']).rename('NDBI');

  // BAI hinzufügen
  var bai = image.expression(
    '1 / ((0.1 - RED)**2 + (0.06 - NIR)**2)', {
      'RED': image.select('SR_B4'),
      'NIR': image.select('SR_B5')
  }).rename('BAI');

  return image.addBands([ndvi, nbr, ndbi, bai]);
}

// Maske auf LA County (verhindert Pixel außerhalb)
function maskToCounty(image, county) {
  var mask = ee.Image.constant(1).clipToCollection(county);
  return image.updateMask(mask);
}


// Prefire und Postfire definieren
var preFire = landsatCollection
.filterDate('2024-12-01', '2025-01-01')
.map(maskClouds)
.map(addIndices)
.mean();

var postFire = landsatCollection
.filterDate('2025-01-31', '2025-03-15')
.map(maskClouds)
.map(addIndices)
.mean();


// Differenzen aller Features berechnen
var diffBands = [
  'SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
  'NDVI','NBR','NDBI', 'BAI'
];

var diff = preFire.select(diffBands)
  .subtract(postFire.select(diffBands))
  .rename([
    'dSR_B1','dSR_B2','dSR_B3','dSR_B4','dSR_B5','dSR_B6','dSR_B7',
    'dNDVI','dNBR','dNDBI', 'dBAI'
  ]);



// drei verschiedene Featurestacks ausprobieren

// Variante A: Postfire-Bänder + Differenzen 
var stackA = maskToCounty( 
  postFire.addBands(diff)
    .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
           'NDVI','NBR','NDBI', 'BAI',
           'dSR_B1','dSR_B2','dSR_B3','dSR_B4','dSR_B5','dSR_B6','dSR_B7',
           'dNDVI','dNBR','dNDBI', 'dBAI']),
  losAngelesCounty
  );

// Variante B: einfach alle Differenzen
var stackB = maskToCounty(
  diff.select(['dSR_B1','dSR_B2','dSR_B3','dSR_B4','dSR_B5','dSR_B6','dSR_B7',
           'dNDVI','dNBR','dNDBI', 'dBAI']),
  losAngelesCounty
  );

// Variante C: Nur Differenz-Indices (ohne Einzelbänder)
var stackC = maskToCounty(
  diff.select(['dNDVI','dNBR','dNDBI', 'dBAI']),
  losAngelesCounty
  );


// Trainingsgebiete aufrufen und mit Klasse versehen 

var water            = table.map(function(f){return f.set('class',4)});
var soil             = table2.map(function(f){return f.set('class',5)});
var settlement       = table3.map(function(f){return f.set('class',3)});
var burnedSettlement = table4.map(function(f){return f.set('class',2)});
var street           = table5.map(function(f){return f.set('class',6)});
var vegetation       = table6.map(function(f){return f.set('class',0)});
var burnedVegetation = table7.map(function(f){return f.set('class',1)});

var trainingPolygons = vegetation.merge(burnedVegetation).merge(burnedSettlement).merge(settlement)
  .merge(water).merge(soil).merge(street);


// Kontrollgebiete aufrufen und mit Klassen versehen
var kontrollevegetation          = table12.map(function(f){return f.set('class',0)});
var kontrolleverbranntevegetation= table14.map(function(f){return f.set('class',1)});
var kontrolleverbranntesiedlung  = table13.map(function(f){return f.set('class',2)});
var kontrollesiedlung            = table10.map(function(f){return f.set('class',3)});
var kontrollewasser              = table15.map(function(f){return f.set('class',4)});
var kontrolleboden               = table9.map(function(f){return f.set('class',5)});
var kontrollestrasse             = table11.map(function(f){return f.set('class',6)});

var controlPolygons = kontrollevegetation.merge(kontrolleverbranntevegetation).merge(kontrolleverbranntesiedlung)
  .merge(kontrollesiedlung).merge(kontrollewasser)
  .merge(kontrolleboden).merge(kontrollestrasse);

//// Auswertung

var classNames = ['Vegetation','Verbrannte Veg.','Verbrannte Siedl.',
                  'Siedlung','Wasser','Boden','Strasse'];
//Klassenfarben definieren
var visParams = {
  min: 0,
  max: 6,
  palette: ['green','red','blue','gray','cyan','yellow','black'],
  forceRgbOutput : true
};

function evaluateStack(stack, name) {

  // Training
  var trainSamples = stack.sampleRegions({
    collection: trainingPolygons,
    properties: ['class'],
    scale: 30
  });

  var clf = ee.Classifier.smileRandomForest(200)
    .train({
      features: trainSamples,
      classProperty: 'class',
      inputProperties: stack.bandNames()
    });

  // Kontrolldaten klassifizieren
  var controlSamples = stack.sampleRegions({
    collection: controlPolygons,
    properties: ['class'],
    scale: 30
  });

  var validated = controlSamples.classify(clf);
  
  var importance = ee.Dictionary(clf.explain().get('importance'));
print('Feature Importance (' + name + '):', importance);

  // Konfusionsmatrix auf Kontrollgebieten
  var cm = validated.errorMatrix('class', 'classification');

  print(name);
  print('Konfusionsmatrix:', cm);
  print('Overall Accuracy:', cm.accuracy());
  print('Kappa:', cm.kappa());
  print('Producers Accuracy (Recall je Klasse):', cm.producersAccuracy());
  print('Consumers Accuracy (Precision je Klasse):', cm.consumersAccuracy());

  // Karte ausgeben für visuelle Inspektion
 var classified = stack.classify(clf);
 
// Flächenberechnung

function calculateArea(image, geometry, classValue) {

  var areaImage = ee.Image.pixelArea()
    .updateMask(image.eq(classValue));

  var area = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geometry,
    scale: 30,
    maxPixels: 1e13
  });

  return ee.Number(area.get('area')).divide(1e6); // m² → km²
}

//Regionen definieren

var regions = {
  'LA': losAngelesCounty.geometry(),
  'Palisades': Palisades,
  'Eaton': Eaton,
};


// Für jede Region: anzeigen + exportieren

Object.keys(regions).forEach(function(regionName) {

var geom = regions[regionName]
  .transform('EPSG:32611', 1);   
  
var clipped = classified
  .clip(geom)
  .updateMask(ee.Image.constant(1).clip(geom));
  // Flächen berechnen
  var burnedVeg = calculateArea(clipped, geom, 1);
  var burnedSett = calculateArea(clipped, geom, 2);

  print('Fläche ' + name + ' | ' + regionName);
  print('Verbrannte Vegetation (km²):', burnedVeg);
  print('Verbrannte Siedlung (km²):', burnedSett);

  // Karte anzeigen
  Map.addLayer(clipped, {
    min:0, max:6,
    palette:['green','red','blue','gray','cyan','yellow','black']
  }, name + ' - ' + regionName);

  // Export
  Export.image.toDrive({
    image: clipped,
    description: name + '_' + regionName,
    scale: 30,
    region: geom,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });

});

  return clf; // Classifier zurückgeben falls du ihn weiter brauchst
}


// Alle drei Stacks auswerten

evaluateStack(stackA, 'Variante A: Postfire + Differenzen');
evaluateStack(stackB, 'Variante B: Nur Differenzen');
evaluateStack(stackC, 'Variante C: Nur Indices');

Map.centerObject(losAngelesCounty, 9);

// zusätzliche Bänder ausgeben für visuelle Inspektion
// Echtfarben RGB 
/*Map.addLayer(postFire, {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],  // Rot, Grün, Blau
  min: 7000,
  max: 20000
}, 'PostFire RGB');

// Falschfarben NIR 
Map.addLayer(postFire, {
  bands: ['SR_B5', 'SR_B4', 'SR_B3'],  // NIR, Rot, Grün
  min: 7000,
  max: 25000
}, 'PostFire Falschfarben NIR');

// SWIR-Komposit 
Map.addLayer(postFire, {
  bands: ['SR_B7', 'SR_B5', 'SR_B3'],  // SWIR2, NIR, Grün
  min: 7000,
  max: 22000
}, 'PostFire SWIR-Komposit');
*/

// 1. Nur Polygone & MultiPolygone behalten
var polygons = Perimeters.map(function(f) {
  var type = f.geometry().type();
  return f.set('gtype', type);
})
.filter(ee.Filter.inList('gtype', ['Polygon', 'MultiPolygon']));

// Fläche berechnen
var withArea = polygons.map(function(f) {
  return f.set('area_m2', f.geometry().area());
});

// sortieren & Top 10
var largest10 = withArea.sort('area_m2', false).limit(10);

// Export
Export.table.toDrive({
  collection: largest10,
  description: 'largest_10_fire_perimeters',
  fileFormat: 'SHP',
  maxVertices: 1e7
});
