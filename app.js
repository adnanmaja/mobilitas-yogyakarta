document.addEventListener('DOMContentLoaded', function() {

    // Add PMTiles protocol
    let protocol = new pmtiles.Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);

    const map = new maplibregl.Map({
        container: 'map',
        style: 'https://tiles.openfreemap.org/styles/dark',
        center: [110.3695, -7.7956],
        zoom: 11,
        pitch: 0,
        bearing: 0
    });

  // Disable rotation for flat feel
  map.dragRotate.disable();
  map.touchZoomRotate.disableRotation();

  // Current state
  let currentLayerType = 'none';
  let currentTimePeriod = 'peak';
  let dropdownPanel = null;

  // All layer groups
  const layerGroups = {
    origins: ['origins-layer-1km', 'origins-layer-300m'],
    destinations: ['destinations-layer-1km', 'destinations-layer-300m'],
    flows: ['peak-flow-layer', 'off-peak-flow-layer', 'weekend-flow-layer'],
    congestion: ['peak-congestion-layer', 'off-peak-congestion-layer', 'weekend-congestion-layer']
  };

  // Time period to layer mapping
  const timeToLayerMap = {
    peak: {
      flows: 'peak-flow-layer',
      congestion: 'peak-congestion-layer'
    },
    'off-peak': {
      flows: 'off-peak-flow-layer',
      congestion: 'off-peak-congestion-layer'
    },
    weekend: {
      flows: 'weekend-flow-layer',
      congestion: 'weekend-congestion-layer'
    }
  };

  // Function to hide all layers except base map
  function hideAllLayers() {
    Object.values(layerGroups).flat().forEach(layerId => {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, 'visibility', 'none');
      }
    });
  }

  // Function to show layers of a specific type
  function showLayerType(layerType) {
    hideAllLayers();
    currentLayerType = layerType;
    
    if (layerType === 'none') {
      return;
    }
    
    if (layerType === 'flows' || layerType === 'congestion') {
      const activeLayerId = timeToLayerMap[currentTimePeriod][layerType];
      if (map.getLayer(activeLayerId)) {
        map.setLayoutProperty(activeLayerId, 'visibility', 'visible');
      }
    } else {
      const layersToShow = layerGroups[layerType];
      layersToShow.forEach(layerId => {
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', 'visible');
        }
      });
    }
  }

  // Function to update time period
  function updateTimePeriod(period) {
    currentTimePeriod = period;
    
    if (currentLayerType === 'flows' || currentLayerType === 'congestion') {
      showLayerType(currentLayerType);
    }
    
    // Update the selected text in dropdown
    const selectedText = document.querySelector('.dropdown-selected');
    if (selectedText) {
      selectedText.textContent = period.charAt(0).toUpperCase() + period.slice(1);
    }
    
    // Update dropdown panel options
    if (dropdownPanel) {
      dropdownPanel.querySelectorAll('.dropdown-panel-option').forEach(option => {
        if (option.dataset.period === period) {
          option.classList.add('active');
        } else {
          option.classList.remove('active');
        }
      });
    }
    
    // Close dropdown panel
    closeDropdownPanel();
  }

  // Function to update layer selection UI
  function updateLayerUI(selectedLayer) {
    document.querySelectorAll('.layer-option').forEach(option => {
      const layer = option.dataset.layer;
      
      if (layer === selectedLayer) {
        option.classList.add('active');
      } else {
        option.classList.remove('active');
      }
    });
    
    const timeControl = document.getElementById('time-control');
    if (selectedLayer === 'flows' || selectedLayer === 'congestion') {
      timeControl.style.display = 'block';
    } else {
      timeControl.style.display = 'none';
      closeDropdownPanel();
    }
  }

  // Function to create floating dropdown panel
  function createDropdownPanel() {
    // Remove existing panel if any
    if (dropdownPanel) {
      dropdownPanel.remove();
    }
    
    // Create new panel
    dropdownPanel = document.createElement('div');
    dropdownPanel.className = 'dropdown-panel';
    dropdownPanel.id = 'time-dropdown-panel';
    
    // Add options
    const periods = [
      { id: 'peak', label: 'Peak' },
      { id: 'off-peak', label: 'Off-Peak' },
      { id: 'weekend', label: 'Weekend' }
    ];
    
    periods.forEach(period => {
      const option = document.createElement('div');
      option.className = 'dropdown-panel-option';
      if (period.id === currentTimePeriod) {
        option.classList.add('active');
      }
      option.dataset.period = period.id;
      option.textContent = period.label;
      
      option.addEventListener('click', () => {
        updateTimePeriod(period.id);
      });
      
      dropdownPanel.appendChild(option);
    });
    
    document.body.appendChild(dropdownPanel);
  }

  // Function to position dropdown panel
  function positionDropdownPanel() {
    if (!dropdownPanel) return;
    
    const dropdownHeader = document.querySelector('.dropdown-header');
    const headerRect = dropdownHeader.getBoundingClientRect();
    
    // Position panel below the dropdown header
    dropdownPanel.style.left = `${headerRect.left}px`;
    dropdownPanel.style.top = `${headerRect.bottom + 5}px`;
  }

  // Function to open dropdown panel
  function openDropdownPanel() {
    const dropdownHeader = document.querySelector('.dropdown-header');
    
    if (!dropdownPanel) {
      createDropdownPanel();
    }
    
    dropdownPanel.classList.add('active');
    dropdownHeader.classList.add('active');
    positionDropdownPanel();
    
    // Close panel when clicking outside
    document.addEventListener('click', closeDropdownOnClickOutside);
  }

  // Function to close dropdown panel
  function closeDropdownPanel() {
    const dropdownHeader = document.querySelector('.dropdown-header');
    
    if (dropdownPanel) {
      dropdownPanel.classList.remove('active');
    }
    
    dropdownHeader.classList.remove('active');
    document.removeEventListener('click', closeDropdownOnClickOutside);
  }

  // Function to handle clicks outside dropdown
  function closeDropdownOnClickOutside(event) {
    const dropdownHeader = document.querySelector('.dropdown-header');
    const timeControl = document.getElementById('time-control');
    
    if (dropdownPanel && !dropdownPanel.contains(event.target) && 
        !dropdownHeader.contains(event.target) &&
        timeControl.style.display !== 'none') {
      closeDropdownPanel();
    }
  }

  // Function to handle panel toggle
  function setupPanelToggle() {
    const panel = document.querySelector('.control-panel');
    const toggleBtn = document.getElementById('panel-toggle');

    toggleBtn.addEventListener('click', () => {
      panel.classList.toggle('panel-is-closed');
      // Close dropdown when panel is toggled
      closeDropdownPanel();
    });
  }

  // Function to handle landing page
  function setupLandingPage() {
    const landingPage = document.getElementById('landing-page');
    const enterMapBtn = document.getElementById('enter-map');
    const learnMoreBtn = document.getElementById('learn-more');
    
    if (enterMapBtn) {
      enterMapBtn.addEventListener('click', () => {
        landingPage.classList.add('hidden');
        map.resize();
        document.getElementById('ui').style.display = 'block';
      });
    }
    
    if (learnMoreBtn) {
      learnMoreBtn.addEventListener('click', () => {
        window.location.href = 'https://github.com/adnanmaja/mobilitas-yogyakarta';
      });
    }
  }

  // Setup event listeners
  function setupEventListeners() {
    // Layer selection buttons
    document.querySelectorAll('.layer-option').forEach(option => {
      option.addEventListener('click', () => {
        const layerType = option.dataset.layer;
        showLayerType(layerType);
        updateLayerUI(layerType);
      });
    });

    // Time dropdown click handler
    const dropdownHeader = document.querySelector('.dropdown-header');
    if (dropdownHeader) {
      dropdownHeader.addEventListener('click', (e) => {
        e.stopPropagation();
        const isActive = dropdownHeader.classList.contains('active');
        
        if (isActive) {
          closeDropdownPanel();
        } else {
          openDropdownPanel();
        }
      });
    }
  }

  // Handle window resize
  window.addEventListener('resize', () => {
    if (dropdownPanel && dropdownPanel.classList.contains('active')) {
      positionDropdownPanel();
    }
  });

  // Initialize map
  map.on('load', () => {
    // ORIGINS
    map.addSource('origins_1000', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/origin_v2_1_1000m.pmtiles',
        attribution: ''
    });
    map.addSource('origins_300', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/origin_v2_1_300m.pmtiles',
        attribution: ''
    });

    map.addLayer({
      id: 'origins-layer-300m',
      type: 'circle',
      source: 'origins_300',
      'source-layer': 'dest_300',
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
          0, '#1a4d6d',
          50, '#4a9fd8',
          100, '#6ec6ff'
        ],
        'circle-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'origin_score'],
          0, 0.3,
          100, 0.7
        ],
        'circle-blur': 0.5
      }
    });

    map.addLayer({
      id: 'origins-layer-1km',
      type: 'circle',
      source: 'origins_1000',
      'source-layer': 'dest_300',
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
          0, '#1a4d6d',
          50, '#4a9fd8',
          100, '#6ec6ff'
        ],
        'circle-opacity': 0.5,
        'circle-blur': 0.2
      }
    });

    // DESTINATIONS
    map.addSource('destinations_1000', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/destination_v3_1000m.pmtiles',
        attribution: ''
    });
    map.addSource('destinations_300', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/destination_v3_300m.pmtiles',
        attribution: ''
    });

    map.addLayer({
      id: 'destinations-layer-300m',
      type: 'circle',
      source: 'destinations_300',
      'source-layer': 'dest_300',
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

    map.addLayer({
      id: 'destinations-layer-1km',
      type: 'circle',
      source: 'destinations_1000',
      'source-layer': 'dest_300',
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
    map.addSource('peak-flow-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/peak_routed_vectors_1000m_edge_flows.pmtiles',
        attribution: ''
    });
    map.addSource('off-peak-flow-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/off_peak_routed_vectors_1000m_edge_flows.pmtiles',
        attribution: ''
    });
    map.addSource('weekend-flow-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/weekendrouted_vectors_1000m_edge_flows.pmtiles',
        attribution: ''
    });

    map.addLayer({
      'id': 'peak-flow-layer',
      'type': 'line',
      'source': 'peak-flow-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, 1.2,
          0.0026, 10
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, '#34d399',
          0.00004, '#fbbf24',
          0.0026, '#ef4444'
        ],
        'line-opacity': 0.95
      }
    });

    map.addLayer({
      'id': 'off-peak-flow-layer',
      'type': 'line',
      'source': 'off-peak-flow-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, 1.2,
          0.0026, 10
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, '#34d399',
          0.00004, '#fbbf24',
          0.0026, '#ef4444'
        ],
        'line-opacity': 0.95
      }
    });

    map.addLayer({
      'id': 'weekend-flow-layer',
      'type': 'line',
      'source': 'weekend-flow-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, 1.2,
          0.0026, 10
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'flow'],
          0.000001, '#34d399',
          0.00004, '#fbbf24',
          0.0026, '#ef4444'
        ],
        'line-opacity': 0.95
      }
    });

    // CONGESTIONS
    map.addSource('peak-congestion-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/peak_routed_vectors_1000m_congestions.pmtiles',
        attribution: ''
    });
    map.addSource('off-peak-congestion-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/off_peak_routed_vectors_1000m_congestions.pmtiles',
        attribution: ''
    });
    map.addSource('weekend-congestion-data', {
        type: 'vector',
        url: 'pmtiles://https://adnanmaja.github.io/mobilitas-yogyakarta/data/weekend_routed_vectors_1000m_congestions.pmtiles',
        attribution: ''
    });

    map.addLayer({
      'id': 'peak-congestion-layer',
      'type': 'line',
      'source': 'peak-congestion-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0, 2,
          2.85, 6
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0.0039, '#2ecc71',
          0.1529, '#f1c40f',
          0.4774, '#e67e22',
          1.1355, '#e74c3c',
          2.0736, '#c0392b',
          2.8561, '#8e44ad'
        ],
        'line-opacity': 0.8
      }
    });

    map.addLayer({
      'id': 'off-peak-congestion-layer',
      'type': 'line',
      'source': 'off-peak-congestion-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0, 2,
          2.85, 6
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0.0039, '#2ecc71',
          0.1529, '#f1c40f',
          0.4774, '#e67e22',
          1.1355, '#e74c3c',
          2.0736, '#c0392b',
          2.8561, '#8e44ad'
        ],
        'line-opacity': 0.8
      }
    });

    map.addLayer({
      'id': 'weekend-congestion-layer',
      'type': 'line',
      'source': 'weekend-congestion-data',
      'source-layer': 'default',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0, 2,
          2.85, 6
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'congestion'],
          0.0039, '#2ecc71',
          0.1529, '#f1c40f',
          0.4774, '#e67e22',
          1.1355, '#e74c3c',
          2.0736, '#c0392b',
          2.8561, '#8e44ad'
        ],
        'line-opacity': 0.8
      }
    });

    // Initialize with only the base map 
    updateLayerUI('none');
    showLayerType('none');

    // Setup event listeners after map loads
    setupEventListeners();
    setupPanelToggle();
    setupLandingPage();
  });
});