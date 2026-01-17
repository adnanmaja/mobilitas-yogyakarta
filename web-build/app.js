// Load token from config
if (typeof CONFIG === 'undefined') {
  console.error('CONFIG not loaded. Make sure to run build script first.');
}

mapboxgl.accessToken = CONFIG.mapboxToken;

let protocol = new pmtiles.Protocol();
mapboxgl.addProtocol("pmtiles", protocol.tile);

const map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/dark-v11',
  center: [110.3695, -7.7956], // Yogyakarta
  zoom: 11,
  pitch: 0,
  bearing: 0
});

// Disable rotation 
map.dragRotate.disable();
map.touchZoomRotate.disableRotation();

map.on('load', () => {

  // ORIGINS 
  map.addSource('origins_1000', {
    type: 'vector',
    url: 'adnanmaja.github.io/mobilitas-yogyakarta/data/origin_v2_1_1000m.pmtiles',
    attribution: ''
  });
  map.addSource('origins_300', {
    type: 'vector',
    url: 'adnanmaja.github.io/mobilitas-yogyakarta/data/origin_v2_1_300m.pmtiles',
    attribution: ''
  });

  map.addLayer({
    id: 'origins-layer-300m',
    type: 'circle',
    source: 'origins_300',
    'source-layer': 'dest_300', // forgot to rename the layers
    minzoom: 12,
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['get', 'origin_score'],
        0, 0,
        100, 18
      ],
      'circle-color': [
        'interpolate',
        ['linear'],
        ['get', 'origin_score'],
        0, '#1a4d6d',      // darker blue for sparse
        50, '#4a9fd8',     // medium blue
        100, '#6ec6ff'     // bright blue for dense
    ],
      'circle-opacity': [
        'interpolate',
        ['linear'],
        ['get', 'origin_score'],
        0, 0.3,      // more transparent for sparse
        100, 0.7     // more opaque for dense
    ],
    'circle-blur': 0.5
    }
  });

  map.addLayer({
    id: 'origins-layer-1km',
    type: 'circle',
    source: 'origins_1000',
    'source-layer': 'dest_300', // forgot to rename the layers
    maxzoom: 12,
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['get', 'origin_score'],
        0, 0,
        100, 20
      ],
      'circle-color': [
        'interpolate',
        ['linear'],
        ['get', 'origin_score'],
        0, '#1a4d6d',      // darker blue for sparse
        50, '#4a9fd8',     // medium blue
        100, '#6ec6ff'     // bright blue for dense
    ],
      'circle-opacity': 0.5,
      'circle-blur': 0.2
    }
  });


  // DESTINATIONS 
  map.addSource('destinations_1000', {
    type: 'vector',
    url: 'adnanmaja.github.io/mobilitas-yogyakarta/data/destination_v3_1000m.pmtiles',
    attribution: ''
  });
  map.addSource('destinations_300', {
    type: 'vector',
    url: 'adnanmaja.github.io/mobilitas-yogyakarta/data/destination_v3_300m.pmtiles',
    attribution: ''
  });


  map.addLayer({
    id: 'destinations-layer-300m',
    type: 'circle',
    source: 'destinations_300',
    'source-layer': 'dest_300', // forgot to rename the layers
    minzoom: 12,
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, 0,
        100, 28
      ],
      'circle-color': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, '#8b3a3a',      // darker red for sparse
        50, '#d66b66',     // medium red
        100, '#ff8a80'     // bright red for dense
    ],
      'circle-opacity': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, 0.3,
        100, 0.7
    ],
    'circle-blur': 0.5
    }
  });

   map.addLayer({
    id: 'destinations-layer-1km',
    type: 'circle',
    source: 'destinations_1000',
    'source-layer': 'dest_300', // forgot to rename the layers
    maxzoom: 12,
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, 0,
        100, 28
      ],
      'circle-color': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, '#8b3a3a',      
        50, '#d66b66',    
        100, '#ff8a80'     
    ],
      'circle-opacity': [
        'interpolate',
        ['linear'],
        ['get', 'destination_score'],
        0, 0.3,
        100, 0.7
    ],
      'circle-blur': 0.5
    }
  });



  // EDGE FLOWS 
  map.addSource('flow-data', {
    type: 'vector',
    data: 'adnanmaja.github.io/mobilitas-yogyakarta/data/routed_vectors2_edge_flows.geojson' 
  });

  map.addLayer({
    'id': 'flow-layer',
    'type': 'line',
    'source': 'flow-data',
    'source-layer': 'dest_300', // forgot to rename the layers
    'layout': {
        'line-join': 'round',
        'line-cap': 'round'
    },
    'paint': {
        'line-width': [
            'interpolate',
            ['linear'],
            ['get', 'flow'],
            0.000001, 1,    // Min flow -> 1px width
            0.0026, 12      // Max flow -> 12px width
        ],
        'line-color': [
            'interpolate',
            ['linear'],
            ['get', 'flow'],
            0.000001, '#34d399', // Low flow: Green
            0.00004, '#fbbf24',  // Average flow: Yellow
            0.0026, '#ef4444'    // High flow: Red
        ],
        'line-opacity': 0.8
    }
  });

  // Start with only origins visible
  map.setLayoutProperty('destinations-layer-1km', 'visibility', 'none');
  map.setLayoutProperty('destinations-layer-300m', 'visibility', 'none');
  map.setLayoutProperty('flow-layer', 'visibility', 'none');
});

document.getElementById('show-origins').onclick = () => {
  map.setLayoutProperty('origins-layer-1km', 'visibility', 'visible');
  map.setLayoutProperty('origins-layer-300m', 'visibility', 'visible');
  map.setLayoutProperty('destinations-layer-1km', 'visibility', 'none');
  map.setLayoutProperty('destinations-layer-300m', 'visibility', 'none');
  map.setLayoutProperty('flow-layer', 'visibility', 'none');
};

document.getElementById('show-destinations').onclick = () => {
map.setLayoutProperty('origins-layer-1km', 'visibility', 'none');
  map.setLayoutProperty('origins-layer-300m', 'visibility', 'none');
  map.setLayoutProperty('destinations-layer-1km', 'visibility', 'visible');
  map.setLayoutProperty('destinations-layer-300m', 'visibility', 'visible');
  map.setLayoutProperty('flow-layer', 'visibility', 'none');
};

document.getElementById('show-flows').onclick = () => {
map.setLayoutProperty('origins-layer-1km', 'visibility', 'none');
  map.setLayoutProperty('origins-layer-300m', 'visibility', 'none');
  map.setLayoutProperty('destinations-layer-1km', 'visibility', 'none');
  map.setLayoutProperty('destinations-layer-300m', 'visibility', 'none');
  map.setLayoutProperty('flow-layer', 'visibility', 'visible');
};
