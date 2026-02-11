document.addEventListener('DOMContentLoaded', function() {

  // Add PMTiles protocol
    let protocol = new pmtiles.Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);

    const MAPTILER_TOKEN = 'MAPTILER_TOKEN';

    const map = new maplibregl.Map({
        container: 'map',
        style: `https://api.maptiler.com/maps/dataviz-dark/style.json?key=${MAPTILER_TOKEN}`,
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
  let currentAmenityType = 'amenity-hbnw'
  let timeDropdownPanel = null;
  let typeDropdownPanel = null;

  // All layer groups
  const layerGroups = {
    residential: ['residential-layer-500m'],
    employment: ['employment-layer-500m'],
    amenity: ['hbnw-amenity-layer-500m', 'nhb-amenity-layer-500m'],
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

  const typeToLayerMap = {
  'amenity-hbnw': 'hbnw-amenity-layer-500m',
  'amenity-nhb': 'nhb-amenity-layer-500m',
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

    // Handle flows and congestion (time-based)
    if (layerType === 'flows' || layerType === 'congestion') {
      const activeLayerId = timeToLayerMap[currentTimePeriod][layerType];
      if (map.getLayer(activeLayerId)) {
        map.setLayoutProperty(activeLayerId, 'visibility', 'visible');
      }
    } 
    // Handle amenity (type-based)
    else if (layerType === 'amenity') {
      const activeLayerId = typeToLayerMap[currentAmenityType];
      if (map.getLayer(activeLayerId)) {
        map.setLayoutProperty(activeLayerId, 'visibility', 'visible');
      }
    }
    // Handle residential and employment (show all layers in group)
    else {
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

  // Function to update amenity type
  function updateAmenityTypes(type) {
    currentAmenityType = type

    if (currentLayerType === 'amenity') {
      showLayerType(currentLayerType);
    }

    const selectedText = document.querySelector('.dropdown-selected');
    if (selectedText) {
      selectedText.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    }

    if (dropdownPanel) {
      dropdownPanel.querySelectorAll('.dropdown-panel-option').forEach(option => {
        if (option.dataset.type === type) {
          option.classList.add('active');
        } else {
          option.classList.remove('active');
        }
      });
    }

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
    const typeControl = document.getElementById('type-control');
    
    if (selectedLayer === 'flows' || selectedLayer === 'congestion') {
      timeControl.style.display = 'block';
      typeControl.style.display = 'none';
      closeAllDropdownPanels();
    } else if (selectedLayer === 'amenity') {
      typeControl.style.display = 'block';
      timeControl.style.display = 'none';
      closeAllDropdownPanels();
    } else {
      timeControl.style.display = 'none';
      typeControl.style.display = 'none';
      closeAllDropdownPanels();
    }
  }

  // Function to create floating dropdown panel for time periods
  function createTimeDropdownPanel() {
    if (timeDropdownPanel) {
      timeDropdownPanel.remove();
    }
    
    timeDropdownPanel = document.createElement('div');
    timeDropdownPanel.className = 'dropdown-panel time-dropdown-panel';
    timeDropdownPanel.id = 'time-dropdown-panel';
    
    const periods = [
      { id: 'peak', label: 'Peak' },
      { id: 'off-peak', label: 'Off-Peak (unavailable)' },
      { id: 'weekend', label: 'Weekend (unavailable)' }
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
      
      timeDropdownPanel.appendChild(option);
    });
    
    document.body.appendChild(timeDropdownPanel);
  }

  // Function to create floating dropdown panel for amenity type
  function createTypeDropdownPanel() {
    if (typeDropdownPanel) {
      typeDropdownPanel.remove();
    }
    
    typeDropdownPanel = document.createElement('div');
    typeDropdownPanel.className = 'dropdown-panel type-dropdown-panel';
    typeDropdownPanel.id = 'type-dropdown-panel';
    
    const types = [
      { id: 'amenity-hbnw', label: 'HBNW Amenities' },
      { id: 'amenity-nhb', label: 'NHB Amenities' },
    ];
    
    types.forEach(type => {
      const option = document.createElement('div');
      option.className = 'dropdown-panel-option';
      if (type.id === currentAmenityType) {
        option.classList.add('active');
      }
      option.dataset.type = type.id;
      option.textContent = type.label;
      
      option.addEventListener('click', () => {
        updateAmenityTypes(type.id);
      });
      
      typeDropdownPanel.appendChild(option);
    });
    
    document.body.appendChild(typeDropdownPanel);
  }

  // Function to position dropdown panel
  function positionTimeDropdownPanel() {
    if (!timeDropdownPanel) return;
    
    const timeDropdownHeader = document.querySelector('#time-control .dropdown-header');
    const headerRect = timeDropdownHeader.getBoundingClientRect();
    
    timeDropdownPanel.style.left = `${headerRect.left}px`;
    timeDropdownPanel.style.top = `${headerRect.bottom + 5}px`;
  }

  function positionTypeDropdownPanel() {
    if (!typeDropdownPanel) return;
    
    const typeDropdownHeader = document.querySelector('#type-control .dropdown-header');
    const headerRect = typeDropdownHeader.getBoundingClientRect();
    
    typeDropdownPanel.style.left = `${headerRect.left}px`;
    typeDropdownPanel.style.top = `${headerRect.bottom + 5}px`;
  }

  // Separate open/close functions
  function openTimeDropdownPanel() {
    const timeDropdownHeader = document.querySelector('#time-control .dropdown-header');
    
    if (!timeDropdownPanel) {
      createTimeDropdownPanel();
    }
    
    timeDropdownPanel.classList.add('active');
    timeDropdownHeader.classList.add('active');
    positionTimeDropdownPanel();
    
    document.addEventListener('click', closeTimeDropdownOnClickOutside);
  }

  function openTypeDropdownPanel() {
    const typeDropdownHeader = document.querySelector('#type-control .dropdown-header');
    
    if (!typeDropdownPanel) {
      createTypeDropdownPanel();
    }
    
    typeDropdownPanel.classList.add('active');
    typeDropdownHeader.classList.add('active');
    positionTypeDropdownPanel();
    
    document.addEventListener('click', closeTypeDropdownOnClickOutside);
  }

  function closeTimeDropdownPanel() {
    const timeDropdownHeader = document.querySelector('#time-control .dropdown-header');
    
    if (timeDropdownPanel) {
      timeDropdownPanel.classList.remove('active');
    }
    
    timeDropdownHeader.classList.remove('active');
    document.removeEventListener('click', closeTimeDropdownOnClickOutside);
  }

  function closeTypeDropdownPanel() {
    const typeDropdownHeader = document.querySelector('#type-control .dropdown-header');
    
    if (typeDropdownPanel) {
      typeDropdownPanel.classList.remove('active');
    }
    
    typeDropdownHeader.classList.remove('active');
    document.removeEventListener('click', closeTypeDropdownOnClickOutside);
  }

  function closeAllDropdownPanels() {
    closeTimeDropdownPanel();
    closeTypeDropdownPanel();
  }

  // Separate click outside handlers
  function closeTimeDropdownOnClickOutside(event) {
    const timeDropdownHeader = document.querySelector('#time-control .dropdown-header');
    
    if (timeDropdownPanel && 
        !timeDropdownPanel.contains(event.target) && 
        !timeDropdownHeader.contains(event.target)) {
      closeTimeDropdownPanel();
    }
  }

  function closeTypeDropdownOnClickOutside(event) {
    const typeDropdownHeader = document.querySelector('#type-control .dropdown-header');
    
    if (typeDropdownPanel && 
        !typeDropdownPanel.contains(event.target) && 
        !typeDropdownHeader.contains(event.target)) {
      closeTypeDropdownPanel();
    }
  }

  // Function to handle panel toggle
  function setupPanelToggle() {
    const panel = document.querySelector('.control-panel');
    const toggleBtn = document.getElementById('panel-toggle');

    toggleBtn.addEventListener('click', () => {
      panel.classList.toggle('panel-is-closed');
      // Close dropdown when panel is toggled
      closeTimeDropdownPanel();
      closeTypeDropdownPanel();
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
        alert('This visualization shows mobility patterns in Yogyakarta, Indonesia. It displays origins, employment, traffic flows, and congestion levels at different times of day.');
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
    const timeDropdownHeader = document.querySelector('#time-control .dropdown-header');
    if (timeDropdownHeader) {
      timeDropdownHeader.addEventListener('click', (e) => {
        e.stopPropagation();
        const isActive = timeDropdownHeader.classList.contains('active');
        
        if (isActive) {
          closeTimeDropdownPanel();
        } else {
          openTimeDropdownPanel();
        }
      });
    }

    // Type dropdown click handler
    const typeDropdownHeader = document.querySelector('#type-control .dropdown-header');
    if (typeDropdownHeader) {
      typeDropdownHeader.addEventListener('click', (e) => {
        e.stopPropagation();
        const isActive = typeDropdownHeader.classList.contains('active');
        
        if (isActive) {
          closeTypeDropdownPanel();
        } else {
          openTypeDropdownPanel();
        }
      });
    }
  }

  // Handle window resize
  window.addEventListener('resize', () => {
    if (timeDropdownPanel && timeDropdownPanel.classList.contains('active')) {
      positionTimeDropdownPanel();
    }
    if (typeDropdownPanel && typeDropdownPanel.classList.contains('active')) {
      positionTypeDropdownPanel();
    }
  });

  // Initialize map
  map.on('load', () => {
    // Residemtials
    map.addSource('residential_500', {
      type: 'vector',
      data: 'pmtiles://./residential_500m.pmtiles'
    });

    map.addLayer({
      id: 'residential-layer-500m',
      type: 'circle',
      source: 'residential_500',
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['get', 'residential_intensity'],
          0, 0,
          100, 18
        ],
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'residential_intensity'],
          0, '#1a4d6d',
          50, '#4a9fd8',
          100, '#6ec6ff'
        ],
        'circle-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'residential_intensity'],
          0, 0.3,
          100, 0.7
        ],
        'circle-blur': 0.5
      }
    });


    // Employments
    map.addSource('employment_500', {
      type: 'vector',
      data: 'pmtiles://./employment_500m.pmtiles'
    });

    map.addLayer({
      id: 'employment-layer-500m',
      type: 'circle',
      source: 'employment_500',
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['get', 'employment_intensity'],
          0, 0,
          100, 28
        ],
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'employment_intensity'],
          0, '#8b3a3a',
          50, '#d66b66',
          100, '#ff8a80'
        ],
        'circle-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'employment_intensity'],
          0, 0.3,
          100, 0.7
        ],
        'circle-blur': 0.5
      }
    });

    // Amenities
    map.addSource('amenity_500', {
      type: 'vector',
      data: 'pmtiles://./services_amenities_500m.pmtiles'
    });

    map.addLayer({
      id: 'hbnw-amenity-layer-500m',
      type: 'circle',
      source: 'amenity_500',
      // minzoom: 16,
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_hbnw_intensity'],
          0, 0,
          100, 18
        ],
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_hbnw_intensity'],
          0, '#1a4d6d',
          50, '#4a9fd8',
          100, '#6ec6ff'
        ],
        'circle-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_hbnw_intensity'],
          0, 0.3,
          100, 0.7
        ],
        'circle-blur': 0.5
      }
    });

    map.addLayer({
      id: 'nhb-amenity-layer-500m',
      type: 'circle',
      source: 'amenity_500',
      // minzoom: 16,
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_nhb_intensity'],
          0, 0,
          100, 18
        ],
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_nhb_intensity'],
          0, '#1a4d6d',
          50, '#4a9fd8',
          100, '#6ec6ff'
        ],
        'circle-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'amenity_nhb_intensity'],
          0, 0.3,
          100, 0.7
        ],
        'circle-blur': 0.5
      }
    });

    // EDGE FLOWS
    map.addSource('peak-flow-data', {
      type: 'vector',
      data: 'pmtiles://./rea_1000m_edge_flows_v3.pmtiles'
    });
    map.addSource('off-peak-flow-data', {
      type: 'vector',
      data: 'n/a'
    });
    map.addSource('weekend-flow-data', {
      type: 'vector',
      data: 'n/a'
    });

    map.addLayer({
      'id': 'peak-flow-layer',
      'type': 'line',
      'source': 'peak-flow-data',
      'layout': {
        'line-join': 'round',
        'line-cap': 'round'
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'total_flow'],
          0.07966477843001485, 1.2,    // Median: Thin line
          1.399054811172391, 4,      // 90th Percentile: Medium line
          37.315249125300056, 12      // Maximum: Very thick line
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'total_flow'],
          0.07966477843001485, '#34d399', // Median: Green
          0.6013697411714708, '#fbbf24',   // 80th: Amber
          1.399054811172391, '#ef4444',   // 90th Percentile: Red
          2.723173461652838, '#7f1d1d',    // 95th: Deep Red
          37.315249125300056, '#780ff1' // Max: Purple
        ],
        'line-opacity': 0.95
      }
    });

    map.addLayer({
      'id': 'off-peak-flow-layer',
      'type': 'line',
      'source': 'off-peak-flow-data',
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
      type: 'geojson',
      data: 'pmtiles://.rea_1000m_congestions_v4.pmtiles'
    });
    map.addSource('off-peak-congestion-data', {
      type: 'geojson',
      data: 'n/a'
    });
    map.addSource('weekend-congestion-data', {
      type: 'vector',
      data: 'n/a'
    });

    map.addLayer({
      'id': 'peak-congestion-layer',
      'type': 'line',
      'source': 'peak-congestion-data',
      'filter': ['>=', ['get', 'vc_ratio'], 0.10470038663396035], // 70th Percentile
      'layout': {
        'line-join': 'round',
        'line-cap': 'round',
        'line-sort-key': ['get', 'vc_ratio']
      },
      'paint': {
        'line-width': [
          'interpolate',
          ['linear'],
          ['get', 'vc_ratio'],
          0.104700386633960358, 1.5,   // 70th Percentile: Thin
          0.4497026527666784, 4,     // 90th Percentile: Medium
          0.8355367868226627, 8,     // 95th Percentile: Thick
          29.087919099787882, 12     // Maximum: Very Thick
        ],
        'line-color': [
          'interpolate',
          ['linear'],
          ['get', 'vc_ratio'],
          0.10470038663396035, '#e67e22', // 70th Percentile: Orange
          0.20012278191347832, '#e74c3c', // 80th Percentile: Red
          0.4497026527666784, '#c0392b', // 90th Percentile: Deep Red
          0.8355367868226627, '#800000', // 95th Percentile: Dark Maroon
          29.087919099787882, '#780ff1'  // Max: Purple
        ],
        'line-opacity': [
          'interpolate',
          ['linear'],
          ['get', 'vc_ratio'],
          0.10470038663396035, 0.5,   // Start visibility at the new 70th percentile
          0.4497026527666784, 1      // Full opacity at 90th percentile
        ]
      }
    });

    map.addLayer({
      'id': 'off-peak-congestion-layer',
      'type': 'line',
      'source': 'off-peak-congestion-data',
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

    // Initialize with base map (none selected)
    updateLayerUI('none');
    showLayerType('none');

    // Setup event listeners after map loads
    setupEventListeners();
    setupPanelToggle();
    setupLandingPage();
  });
}); // End of DOMContentLoaded