// =================================================================
// 1. CONFIGURATION
// =================================================================
// *** This is the only thing you change to switch base water classifiers ***
// Valid keys: 'MUwi_SnowRefined', 'WI2015_SnowRefined', 'MuWI_SnowRefined',
//             'Zou2018_SnowRefined', 'Jones2019', 'Zou2018'
var watermethod = 'MUwi_SnowRefined';

var area        = '7';
var start_date  = '2017-01-01';
var end_date    = '2025-01-01';
var batchSize   = 500;
var pageNumber  = 4;

// Asset references (confirmed project: ee-ucterrestrialhydrology)
var biasAsset  = ee.FeatureCollection("projects/ee-ucterrestrialhydrology/assets/Bias_result_USGS");
var fc         = ee.FeatureCollection('projects/ee-ucterrestrialhydrology/assets/USGS_Basin_fin_' + area);
var orthoAsset = ee.FeatureCollection("projects/ee-ucterrestrialhydrology/assets/Orthogonal_preliminary_updated_1to1_mapped");

// STEP A: Extract unique site_no values present in the Bias asset
var uniqueSites = biasAsset.aggregate_array('site_no').distinct();

// STEP B: Filter orthoAsset to those sites
var globalMatches = orthoAsset.filter(ee.Filter.inList('site_no', uniqueSites));

// STEP C: Restrict to COMIDs belonging to the target area
var areaNum = ee.Number.parse(area);
var regionalMatches = globalMatches.filter(ee.Filter.and(
  ee.Filter.gte('COMID', areaNum.multiply(1e7)),
  ee.Filter.lt('COMID', areaNum.add(1).multiply(1e7))
)).sort('COMID');

// STEP D: Paging — select one batch of COMIDs
var startIdx         = (pageNumber - 1) * batchSize;
var selectedFeatures = ee.FeatureCollection(regionalMatches.toList(batchSize, startIdx));
var comids           = selectedFeatures.aggregate_array('COMID');

// STEP E: Filter basin asset to the selected COMIDs
var rp    = fc.filter(ee.Filter.inList('COMID', comids));
var arcrw = rp;

// =================================================================
// 2. IMAGE COLLECTION
// =================================================================
// B1 (UltraBlue, 60m) is always included even though only the MUwi
// classifier uses it — harmless extra band for the other methods,
// and required so the same image collection works for every watermethod.
function merge_collections_std_bandnames_collection1tier1() {
  var bnst = ['B1', 'B2', 'B3', 'B4', 'B11', 'QA60', 'B8', 'B12'];
  var bns  = ['UltraBlue', 'Blue', 'Green', 'Red', 'Swir1', 'BQA', 'Nir', 'Swir2'];
  return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(bnst, bns);
}

// =================================================================
// 3. HELPER FUNCTIONS
// =================================================================

// --- Cloud masking (QA60 unpacking) ---
var Unpack = function(qualityBand, startingBit, bitWidth) {
  return qualityBand.rightShift(startingBit).bitwiseAnd(
    ee.Number(2).pow(bitWidth).subtract(1).int()
  );
};

var UnpackAll = function(bitBand) {
  var bitInfoTOA = { 'Cloud': [10, 1] };
  var unpackedImage = ee.Image();
  for (var key in bitInfoTOA) {
    unpackedImage = ee.Image.cat(
      unpackedImage,
      Unpack(bitBand, bitInfoTOA[key][0], bitInfoTOA[key][1]).rename(key)
    );
  }
  return unpackedImage.select(Object.keys(bitInfoTOA));
};

var ClassifyWaterFmask = function(image) {
  var ndvi = image.normalizedDifference(['Nir', 'Red']);
  var nir  = image.select(['Nir']).multiply(0.0001);
  return ndvi.lt(0.01).and(nir.lt(0.11)).or(ndvi.lt(0.1).and(nir.lt(0.05)));
};

exports.AddFmask = function(image) {
  var temp   = UnpackAll(image.select(['BQA']));
  var fwater = ClassifyWaterFmask(image);
  var fmask  = fwater.rename(['fmask'])
    .where(temp.select(['Cloud']), ee.Image(4))
    .mask(temp.select(['Cloud']).gte(0));
  return image.addBands(fmask);
};

// --- Spectral indices ---
exports.Ndvi  = function(image) { return image.normalizedDifference(['Nir', 'Red']).rename('ndvi'); };
exports.Mndwi = function(image) { return image.normalizedDifference(['Green', 'Swir1']).rename('mndwi'); };
exports.Mbsrv = function(image) { return image.select(['Green']).add(image.select(['Red'])).rename('mbsrv'); };
exports.Mbsrn = function(image) { return image.select(['Nir']).add(image.select(['Swir1'])).rename('mbsrn'); };
exports.Evi   = function(image) {
  return image.expression(
    '2.5 * (Nir - Red) / (1 + Nir + 6 * Red - 7.5 * Blue)', {
      'Nir':  image.select(['Nir']).multiply(0.0001),
      'Red':  image.select(['Red']).multiply(0.0001),
      'Blue': image.select(['Blue']).multiply(0.0001)
    }
  ).rename(['evi']);
};
exports.Awesh = function(image) {
  return image.expression(
    'Blue + 2.5 * Green + (-1.5) * mbsrn + (-0.25) * Swir2', {
      'Blue':  image.select(['Blue']),
      'Green': image.select(['Green']),
      'mbsrn': exports.Mbsrn(image).select(['mbsrn']),
      'Swir2': image.select(['Swir2'])
    }
  );
};

// =================================================================
// 4. SNOW GUARD, SEDIMENT GUARD, AND BASE WATER CLASSIFIERS
// =================================================================
// This section is the point of the whole refactor:
//   - ClassifySnow + ApplyGuards are IDENTICAL for every method (verified
//     byte-for-byte across all 4 original scripts). They live here ONCE.
//   - Each base classifier below only computes "initial water" — plain
//     water/no-water, no snow or sediment handling. ApplyGuards is what
//     turns any of them into a "*_SnowRefined" method.
//   - BASE_WATER_CLASSIFIERS maps a watermethod string straight to a
//     function reference — this is the "variable name points at the
//     function" mechanism you asked for.

// --- Tightened snow classification with SWIR1 ceiling (Guard 1) ---
var SWIR1_SNOW_CEILING = 0.15;

exports.ClassifySnow = function(image) {
  var imgScaled = image.divide(10000);
  var ndsi = imgScaled.normalizedDifference(['Green', 'Swir1']).rename('NDSI');
  var ndvi = imgScaled.normalizedDifference(['Nir', 'Red']).rename('NDVI');

  var isTrulyLowSwir1 = imgScaled.select('Swir1').lt(SWIR1_SNOW_CEILING);

  var standardSnow = ndsi.gte(0.4)
                         .and(imgScaled.select('Nir').gt(0.11))
                         .and(isTrulyLowSwir1);

  var forestSnow   = ndvi.gt(0.1)
                         .and(ndsi.gt(0.1))
                         .and(ndsi.lt(0.4))
                         .and(isTrulyLowSwir1);

  var passDarkTarget = imgScaled.select('Green').gte(0.10);

  return standardSnow.or(forestSnow).and(passDarkTarget).rename('isSnow');
};

// --- Sediment guard thresholds (Guard 2) ---
var MNDWI_WATER_THRESHOLD = 0.05;
var NIR_SNOW_FLOOR        = 0.20;

// --- Generic guard wrapper: initialWater -> refinedWater ---
// Any base classifier below (or a new one you add later) becomes a
// "SnowRefined" method just by passing its initialWater mask through this.
exports.ApplyGuards = function(image, initialWater) {
  var imgScaled     = image.divide(10000);
  var mndwiScaled    = imgScaled.normalizedDifference(['Green', 'Swir1']);
  var isSedimentProtected = initialWater
    .and(mndwiScaled.gt(MNDWI_WATER_THRESHOLD))
    .and(imgScaled.select('Nir').lt(NIR_SNOW_FLOOR));

  var isSnow = exports.ClassifySnow(image);

  return isSedimentProtected
    .or(initialWater.and(isSnow.not()))
    .rename('refinedWater');
};

// --- Base classifier: MUwi (MNDWI using UltraBlue B1) ---
exports.BaseWater_MUwi = function(image) {
  var ub_resampled = image.select('UltraBlue').resample('bilinear');
  var twoband      = ub_resampled.rename('UltraBlue').addBands(image.select('Swir2'));
  return twoband.normalizedDifference(['UltraBlue', 'Swir2']).gt(0).rename('initialWater');
};

// --- Base classifier: WI2015 ---
exports.BaseWater_WI2015 = function(image) {
  var imgScaled = image.divide(10000);
  var wi2015 = imgScaled.expression(
    '1.7204 + 171 * Green + 3 * Red - 70 * Nir - 45 * Swir1 - 71 * Swir2', {
      'Green': imgScaled.select('Green'),
      'Red':   imgScaled.select('Red'),
      'Nir':   imgScaled.select('Nir'),
      'Swir1': imgScaled.select('Swir1'),
      'Swir2': imgScaled.select('Swir2')
    }
  ).rename('WI2015');
  return wi2015.gt(0).rename('initialWater');
};

// --- Base classifier: Zou2018 (MNDWI/NDVI/EVI, raw DN — consistent with original) ---
exports.BaseWater_Zou2018 = function(image) {
  var mndwi = exports.Mndwi(image);
  var ndvi  = exports.Ndvi(image);
  var evi   = exports.Evi(image);
  return (mndwi.gt(ndvi).or(mndwi.gt(evi))).and(evi.lt(0.1)).rename('initialWater');
};

// --- Base classifier: MuWI-C (14-term normalized-difference expression) ---
exports.BaseWater_MuWI = function(image) {
  var imgScaled = image.divide(10000);
  var nd = function(i, j) { return imgScaled.normalizedDifference([i, j]); };
  var muwic = imgScaled.expression(
    '-16.4 * nd23 - 6.9 * nd24 - 8.2 * nd28 - 8.8 * nd211 ' +
    '+ 9.6 * nd212 + 10.8 * nd38 + 6.1 * nd311 + 13.6 * nd312 ' +
    '- 0.28 * nd48 - 3.9 * nd411 - 2.1 * nd412 - 5.3 * nd811 ' +
    '- 5.3 * nd812 - 5.3 * nd1112 - 0.33', {
      'nd23':   nd('Blue',  'Green'),
      'nd24':   nd('Blue',  'Red'),
      'nd28':   nd('Blue',  'Nir'),
      'nd211':  nd('Blue',  'Swir1'),
      'nd212':  nd('Blue',  'Swir2'),
      'nd38':   nd('Green', 'Nir'),
      'nd311':  nd('Green', 'Swir1'),
      'nd312':  nd('Green', 'Swir2'),
      'nd48':   nd('Red',   'Nir'),
      'nd411':  nd('Red',   'Swir1'),
      'nd412':  nd('Red',   'Swir2'),
      'nd811':  nd('Nir',   'Swir1'),
      'nd812':  nd('Nir',   'Swir2'),
      'nd1112': nd('Swir1', 'Swir2')
    }
  ).rename('MuWI_C');
  return muwic.gt(0).rename('initialWater');
};

// --- Jones2019 / DSWE (kept as a standalone, unguarded option — matches original behavior) ---
exports.ClassifyWater_Jones = function(image) {
  var Dswe = function(i) {
    var mndwi = exports.Mndwi(i);
    var mbsrv = exports.Mbsrv(i);
    var mbsrn = exports.Mbsrn(i);
    var awesh = exports.Awesh(i);
    var swir1 = i.select(['Swir1']);
    var nir   = i.select(['Nir']);
    var ndvi  = exports.Ndvi(i);
    var blue  = i.select(['Blue']);
    var swir2 = i.select(['Swir2']);
    var t1 = mndwi.gt(0.124);
    var t2 = mbsrv.gt(mbsrn);
    var t3 = awesh.gt(0);
    var t4 = mndwi.gt(-0.44).and(swir1.lt(900)).and(nir.lt(1500)).and(ndvi.lt(0.7));
    var t5 = mndwi.gt(-0.5).and(blue.lt(1000)).and(swir1.lt(3000)).and(swir2.lt(1000)).and(nir.lt(2500));
    var t  = t1.add(t2.multiply(10)).add(t3.multiply(100)).add(t4.multiply(1000)).add(t5.multiply(10000));
    var noWater  = t.eq(0).or(t.eq(1)).or(t.eq(10)).or(t.eq(100)).or(t.eq(1000));
    var hWater   = t.eq(1111).or(t.eq(10111)).or(t.eq(11011)).or(t.eq(11101)).or(t.eq(11110)).or(t.eq(11111));
    var mWater   = t.eq(111).or(t.eq(1011)).or(t.eq(1101)).or(t.eq(1110)).or(t.eq(10011)).or(t.eq(10101)).or(t.eq(10110)).or(t.eq(11001)).or(t.eq(11010)).or(t.eq(11100));
    var pWetland = t.eq(11000);
    var lWater   = t.eq(11).or(t.eq(101)).or(t.eq(110)).or(t.eq(1001)).or(t.eq(1010)).or(t.eq(1100)).or(t.eq(10000)).or(t.eq(10001)).or(t.eq(10010)).or(t.eq(10100));
    var iDswe = noWater.multiply(0).add(hWater.multiply(1)).add(mWater.multiply(2)).add(pWetland.multiply(3)).add(lWater.multiply(4));
    return iDswe.rename('dswe');
  };
  var dswe = Dswe(image);
  return dswe.eq(1).or(dswe.eq(2));
};

// =================================================================
// WATER METHOD REGISTRY
// =================================================================
// This is the "variable name points to the function" mechanism.
// `watermethod` (set in section 1) is just a key into this object.
// Adding a 5th base classifier later = one BaseWater_X function above
// + one line here. No other file logic needs to change.
var WATER_METHODS = {
  'MUwi_SnowRefined': {
    classify: exports.BaseWater_MUwi,
    guarded:  true,
    label:    'MUwiSnowRefined'
  },
  'WI2015_SnowRefined': {
    classify: exports.BaseWater_WI2015,
    guarded:  true,
    label:    'WI2015SnowRefined'
  },
  'MuWI_SnowRefined': {
    classify: exports.BaseWater_MuWI,
    guarded:  true,
    label:    'MuWISnowRefined'
  },
  'Zou2018_SnowRefined': {
    classify: exports.BaseWater_Zou2018,
    guarded:  true,
    label:    'Zou2018SnowRefined'
  },
  'Jones2019': {
    classify: exports.ClassifyWater_Jones,
    guarded:  false,
    label:    'Jones2019'
  },
  'Zou2018': {
    classify: exports.BaseWater_Zou2018,
    guarded:  false,
    label:    'Zou2018'
  }
};

// =================================================================
// 5. RIVER EXTRACTION HELPERS
// =================================================================

exports.GetCenterline = function(clDataset, bound) {
  return clDataset.filterBounds(bound);
};

exports.ExtractChannel = function(image, centerline, bound, maxDistance) {
  var cost = image.not().cumulativeCost({
    source: ee.Image().toByte().paint(centerline, 1).and(image),
    maxDistance: maxDistance,
    geodeticDistance: false
  });
  var channelMask = cost.eq(0).unmask(0).clip(bound).rename(['channelMask']);
  return image.mask(channelMask).unmask(0);
};

exports.RemoveIsland = function(channel, FILL_SIZE) {
  var fill = channel.not().mask(channel.not()).connectedPixelCount(FILL_SIZE).lt(FILL_SIZE).unmask(0);
  return channel.where(fill, ee.Image(1)).rename(['riverMask']);
};

exports.RemoveGeometry = function(f) { return ee.Feature(f).setGeometry(null); };

exports.CalcHillShades = function(image) {
  var mergedDEM     = ee.Image("users/eeProject/MERIT").clip(image.geometry());
  var shiftDistance = 30;
  var dp1 = ee.Image.cat(ee.Image(shiftDistance), ee.Image(shiftDistance));
  return ee.Terrain.hillshade(
    mergedDEM.displace(dp1),
    ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')).add(360),
    image.get('MEAN_SOLAR_ZENITH_ANGLE')
  ).rename(['hillshade']);
};

// =================================================================
// 6. CORE WIDTH LOGIC (GetWidth)
// =================================================================

exports.GetWidth = function(clAngleNorm, riverWithCloud, endInfo, crs, bound, scale, sceneID, note) {

  var GetXsectionEnds = function(f) {
    var utmPrj   = crs;
    var f_prj    = ee.Feature(f).transform(utmPrj);
    var xc       = ee.Number(f_prj.geometry().coordinates().get(0));
    var yc       = ee.Number(f_prj.geometry().coordinates().get(1));
    var orthRad  = ee.Number(f.get('Angle')).divide(180).multiply(Math.PI);
    var width    = ee.Number(f.get('meritwth')).max(f.get('mwth_mean')).max(f.get('gwth_mean')).max(f.get('Awidth')).max(90);
    var halfWidth = width.multiply(1.5);
    var cosRad   = halfWidth.multiply(orthRad.cos());
    var sinRad   = halfWidth.multiply(orthRad.sin());
    var p1 = ee.Geometry.Point([xc.add(cosRad),      yc.add(sinRad)],      utmPrj);
    var p2 = ee.Geometry.Point([xc.subtract(cosRad), yc.subtract(sinRad)], utmPrj);
    return ee.Feature(ee.Geometry.MultiPoint([p1, p2]), {
      'riverID':    f.get('ID_unique'),
      'xc': xc, 'yc': yc,
      'lon': f.get('lon'), 'lat': f.get('lat'),
      'orthRad':    orthRad,
      'MLength':    halfWidth.multiply(2),
      'p1': p1, 'p2': p2,
      'crs': crs, 'sceneID': sceneID, 'note': note, 'utmPrj': utmPrj,
      'width_grwl': f.get('gwth_mean')
    });
  };

  var SwitchGeometry = function(f) {
    return f.setGeometry(ee.Geometry.LineString([f.get('p1'), f.get('p2')]).buffer(30))
            .set('p1', null).set('p2', null);
  };
  var ResetGeometry = function(f) {
    return f.setGeometry(ee.Geometry.Point([f.get('lon'), f.get('lat')]));
  };

  var xsectionsEnds = clAngleNorm.map(GetXsectionEnds);
  var endStat = endInfo.reduceRegions({
    collection: xsectionsEnds,
    reducer: ee.Reducer.anyNonZero()
             .combine(ee.Reducer.sum(),   null, true)
             .combine(ee.Reducer.count(), null, true),
    scale: scale,
    crs: crs
  });
  endStat = endStat.map(function(f) {
    var bankHits = ee.Number(f.get('count')).subtract(ee.Number(f.get('sum')));
    return f.set('count', bankHits);
  });

  var xsections1      = endStat.map(SwitchGeometry);
  var combinedReducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true);
  var xsections = riverWithCloud.reduceRegions({
    collection: xsections1,
    reducer: combinedReducer,
    scale: scale,
    crs: crs
  });

  return xsections.map(ResetGeometry);
};

// =================================================================
// 7. MAIN EXECUTION FUNCTION
// =================================================================

function rw_point(image, aoi, watermethod) {
  var methodConfig = WATER_METHODS[watermethod];
  if (!methodConfig) {
    throw new Error('Unknown watermethod: "' + watermethod + '". Valid keys: ' + Object.keys(WATER_METHODS).join(', '));
  }

  var note        = methodConfig.label + (methodConfig.guarded ? ' Snow-Refined + Sediment Guard' : ' (unguarded)');
  var MAXDISTANCE = 800;
  var FILL_SIZE   = 333;
  var scale       = 10;
  var grwl = ee.FeatureCollection(arcrw);
  var iid  = image.get('GRANULE_ID');

  image = exports.AddFmask(image);

  var crs = image.select(['Green']).projection().crs();

  var fmask     = image.select(['fmask']);
  var mergedDEM = ee.Image("users/eeProject/MERIT").clip(aoi);
  var hillShade = exports.CalcHillShades(image);

  // Base water classification (variable-selected function) + shared guards.
  var baseWater = methodConfig.classify(image);
  var water = methodConfig.guarded ? exports.ApplyGuards(image, baseWater) : baseWater.rename('refinedWater');

  var cl      = exports.GetCenterline(grwl, aoi);
  var channel = exports.ExtractChannel(water, cl, aoi, MAXDISTANCE);
  var river   = exports.RemoveIsland(channel, FILL_SIZE);

  var clAngleNorm = grwl.filterBounds(aoi);
  var endInfo     = river.rename(['river']).clip(water.geometry());

  var exportInfo = river.rename(['river'])
    .addBands(fmask.eq(1).rename(['fwater']))
    .addBands(fmask.eq(2).or(fmask.eq(4)).rename(['fcloud']))
    .addBands(fmask.eq(3).rename(['fsnow']))
    .addBands(mergedDEM.rename(['elevation']))
    .addBands(hillShade.rename(['hillShade']));

  var riverWidths = exports.GetWidth(clAngleNorm, exportInfo, endInfo, crs, aoi, scale, iid, note);
  var collection  = riverWidths.map(exports.RemoveGeometry);

  var meanWidth = collection
    .filterMetadata('any',         'not_equals', -9999)
    .filterMetadata('any',         'equals',     0)
    .filterMetadata('count',       'equals',     2)
    .filterMetadata('fsnow_mean',  'equals',     0)
    .filterMetadata('fcloud_mean', 'equals',     0)
    .map(function(f) {
      var fwidth = ee.Number(15).multiply(Math.PI).add(f.get('MLength')).multiply(f.get('river_mean'));
      return f.set({
        'width':       fwidth,
        'date':        image.date().format('YYYY-MM-dd'),
        'watermethod': watermethod,
        'sceneID':     image.get('GRANULE_ID'),
        'SENSOR_ID':   image.get('SPACECRAFT_NAME'),
        'width_grwl':  f.get('width_grwl')
      });
    });

  return ee.FeatureCollection(meanWidth);
}

// =================================================================
// 8. RUN & EXPORT
// =================================================================

var riverWidthPerImage = function(image) {
  return rw_point(image, image.geometry(), watermethod);
};

var merged     = merge_collections_std_bandnames_collection1tier1();
var merged_fil = merged
  .filterDate(start_date, end_date)
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 25)
  .filterBounds(rp.union().geometry().bounds());

var output     = merged_fil.map(riverWidthPerImage).flatten();
var new_output = output.filter(ee.Filter.gt('width', 0));

// Export filename is derived from the watermethod variable via the
// registry's `label` field — change `watermethod` above and this follows.
var activeMethod = WATER_METHODS[watermethod];
var exportName = activeMethod.label
  + (activeMethod.guarded ? '_SedimentGuard' : '')
  + '_S2_Width_Area' + area + '_Page' + pageNumber + '_' + start_date + '_' + end_date;

Export.table.toDrive({
  collection:     new_output,
  description:    exportName,
  folder:         'USGS_Width_Validation',
  fileFormat:     'csv',
  fileNamePrefix: exportName
});

print('Water Method:', watermethod);
print('Sediment guard SWIR1 ceiling:', SWIR1_SNOW_CEILING);
print('Sediment guard MNDWI threshold:', MNDWI_WATER_THRESHOLD);
print('Sediment guard NIR snow floor:', NIR_SNOW_FLOOR);
print('Exporting for Area:', area);
print('Page Number:', pageNumber);
print('Number of COMIDs in this batch:', comids.size());