<template>
  <div class="flex home">
    <Header />
    <div class="w-full p-10 py-4">
      <div class="pb-4">
        <p class="font-bold text-first-700 text-xl">SHIPMENTS DETAILS</p>
      </div>
      <hr />
      <div class="">
        <div
          to="/shipmentdetails"
          class="block py-8 px-10 bg-white rounded-lg"
        >
          <div class="flex justify-between">
            <div class="flex">
              <svg
                class="self-center h-10 w-10 mr-6"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 640 512"
              >
                <!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. -->
                <path
                  fill="#063E3F"
                  d="M368 0C394.5 0 416 21.49 416 48V96H466.7C483.7 96 499.1 102.7 512 114.7L589.3 192C601.3 204 608 220.3 608 237.3V352C625.7 352 640 366.3 640 384C640 401.7 625.7 416 608 416H576C576 469 533 512 480 512C426.1 512 384 469 384 416H256C256 469 213 512 160 512C106.1 512 64 469 64 416H48C21.49 416 0 394.5 0 368V48C0 21.49 21.49 0 48 0H368zM416 160V256H544V237.3L466.7 160H416zM160 368C133.5 368 112 389.5 112 416C112 442.5 133.5 464 160 464C186.5 464 208 442.5 208 416C208 389.5 186.5 368 160 368zM480 464C506.5 464 528 442.5 528 416C528 389.5 506.5 368 480 368C453.5 368 432 389.5 432 416C432 442.5 453.5 464 480 464z"
                />
              </svg>
              <div class="">
                <p
                  class="
                    self-center
                    text-lg
                    font-bold
                    tracking-tight
                    text-first-700
                  "
                >
                  Shipment Route A
                </p>
                <p class="text-md text-gray-500">
                  {{this.stringShipments()}}
                </p>
              </div>
            </div>
            <p class="font-bold text-lg text-first-700">13 April 2022</p>
          </div>
        </div>
      </div>
      <div class="px-10">
        <div id="map" class="w-full h-96 px-26">
      </div>

        
      </div>
      <div class="pt-6 px-10 text-gray-500 font-semibold">
        <p class="font-bold text-xl text-first-700">Details</p>
        <div class="grid grid-cols-2 w-full">
        <p class="">Total emission </p>
        <p class="text-first-200">1000 kgCO2eq </p>
        <p>Number of Shops Travelled</p>
        <p class="text-first-200">5</p>
        <p>Total Supplies Weight </p>
        <p class="text-first-200">2 Tonnes</p>
        </div>

      </div>
      <div class="flex flex-col pt-6 px-10">
        <p class="font-bold text-xl text-first-700 mb-4">Timeline</p>
        <ol class="relative border-l border-first-700 dark:border-gray-700 left-2 top-2">
          <li v-for="shipment, index in shipments" :key="index" class="mb-10 ml-6">
            <span
              class="
                flex
                absolute
                -left-3
                justify-center
                items-center
                w-6
                h-6
                bg-gray-200
                rounded-full
                ring-2 ring-first-700
                dark:ring-gray-900 dark:bg-blue-900
              "
            >
              <svg
                class="w-3 h-3 text-first-700"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill-rule="evenodd"
                  d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z"
                  clip-rule="evenodd"
                ></path>
              </svg>
            </span>
            <h3
              class="mb-1 text-lg font-semibold text-first-700"
              v-if="index === 0"
            >
              From {{shipment.shop_name.toUpperCase()}}
            </h3>
            <h3
              class="mb-1 text-lg font-semibold text-first-700"
              v-else
            >
              To {{shipment.shop_name.toUpperCase()}}
            </h3>
            <time
              class="
                block
                mb-2
                text-sm
                font-normal
                leading-none
                text-gray-400
                dark:text-gray-500
              "
              >Depart at 08.00AM</time
            >
            <p class="text-base font-normal text-gray-500 dark:text-gray-400">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla ultrices lorem a vehicula imperdiet. 
            </p>
          </li>
        </ol>
      </div>
    </div>
  </div>
</template>

<script>
import 'mapbox-gl/dist/mapbox-gl.css';
import mapboxgl from 'mapbox-gl';
import axios from 'axios'

export default {
  
  transition: "home",
  name: "Shipments",
  asyncData(){

  },
  data() {
    return {
      accessToken: 'pk.eyJ1Ijoic3RlZmFucmFmYWVsIiwiYSI6ImNsMWV2MG1vdTA2YXkzZG54dXBpZGJtd3IifQ.dFkQvD8fs51PMPrFOvaKKg',
      shipments: [
        {"shop_name": "Warehouse", "lat": 22.34614181, "long": 114.1243497},
        {"shop_name": "Shop_27", "lat": 22.2830891, "long": 114.1365621},
        {"shop_name": "Shop_1", "lat": 22.2487869, "long": 114.1538679},
        {"shop_name": "Shop_10", "lat": 22.2628253, "long": 114.2498465},
        {"shop_name": "Shop_46", "lat": 22.3226535, "long": 114.2493481},
        {"shop_name": "Shop_41", "lat": 22.3329969, "long": 114.1670193}
        ],
      message: "",
      pages: [1, 2, 3, 4, 5],
      routes: [],
      routePlan: {'date': '4/5/2022',
        'n_vehicles': 4,
        'routes': [[{'shop_name': 'Warehouse','lat': 22.34614181,'long': 114.1243497},
          {'shop_name': 'Shop_18', 'lat': 22.4940569, 'long': 114.1268154},
          {'shop_name': 'Shop_15', 'lat': 22.3780638, 'long': 114.1297481}],
          [{'shop_name': 'Warehouse', 'lat': 22.34614181, 'long': 114.1243497},
          {'shop_name': 'Shop_27', 'lat': 22.2830891, 'long': 114.1365621},
          {'shop_name': 'Shop_1', 'lat': 22.2487869, 'long': 114.1538679},
          {'shop_name': 'Shop_10', 'lat': 22.2628253, 'long': 114.2498465},
          {'shop_name': 'Shop_46', 'lat': 22.3226535, 'long': 114.2493481},
          {'shop_name': 'Shop_41', 'lat': 22.3329969, 'long': 114.1670193}],
          [{'shop_name': 'Warehouse', 'lat': 22.34614181, 'long': 114.1243497},
          {'shop_name': 'Shop_32', 'lat': 22.3274847, 'long': 114.1621195},
          {'shop_name': 'Shop_37', 'lat': 22.2878544, 'long': 114.1916345},
          {'shop_name': 'Shop_31', 'lat': 22.290093, 'long': 114.19668},
          {'shop_name': 'Shop_50', 'lat': 22.28651139, 'long': 114.1965985},
          {'shop_name': 'Shop_47', 'lat': 22.349888, 'long': 114.202634},
          {'shop_name': 'Shop_6', 'lat': 22.3313068, 'long': 114.1926345},
          {'shop_name': 'Shop_24', 'lat': 22.3147882, 'long': 114.1700475},
          {'shop_name': 'Shop_21', 'lat': 22.208483, 'long': 114.029561}],
          [{'shop_name': 'Warehouse', 'lat': 22.34614181, 'long': 114.1243497},
          {'shop_name': 'Shop_5', 'lat': 22.3239569, 'long': 114.1693153},
          {'shop_name': 'Shop_40', 'lat': 22.335692, 'long': 114.202563},
          {'shop_name': 'Shop_17', 'lat': 22.404843, 'long': 114.224048},
          {'shop_name': 'Shop_54', 'lat': 22.4212369, 'long': 114.2318519},
          {'shop_name': 'Shop_53', 'lat': 22.414411, 'long': 113.9766664},
          {'shop_name': 'Shop_7', 'lat': 22.4077228, 'long': 113.9690563},
          {'shop_name': 'Shop_14', 'lat': 22.3499824, 'long': 114.1000403}]],
        'total_expected_co2': 5.886227328}
      };
  },
  async mounted(){
    // this.createMap();
    await this.getRoute(this.shipments);
  },
  created(){
    this.routes = this.routePlan.routes[0]
    console.log(this.routes)
    
  },
  methods: {
    stringShipments() {
      var strShip = '';
      this.shipments.map((r, index) => {
        if (index+1 == this.shipments.length) {
          strShip += `${r.shop_name}`
        } else {
          strShip += `${r.shop_name}  ->  `
        }
      })
      return strShip;
    },
    async getRoute(waypoints){

      mapboxgl.accessToken = 'pk.eyJ1Ijoic3RlZmFucmFmYWVsIiwiYSI6ImNsMWV2MG1vdTA2YXkzZG54dXBpZGJtd3IifQ.dFkQvD8fs51PMPrFOvaKKg';

      const map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v11',
        center: [114.1243497, 22.34614181],
        zoom: 10
      });

      var wp = ''
      waypoints.map((r, idx) => {
        if(idx == waypoints.length - 1){
          wp += `${r.long},${r.lat}`
        } else {
          wp += `${r.long},${r.lat};`
        }
        const marker = new mapboxgl.Marker({
          color: '#063E3F',
          scale: 0.85
        })
        .setLngLat([r.long,r.lat])
        .addTo(map);
      })

      console.log(wp)

      var url = `https://api.mapbox.com/directions/v5/mapbox/driving/${wp}?alternatives=true&annotations=duration%2Cdistance&geometries=geojson&language=en&overview=full&steps=true&access_token=${mapboxgl.accessToken}`

      const query = await axios.get(url);
      const json = query.data.routes[0].geometry;
      console.log(json.coordinates) 

      const geojson = {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: json.coordinates
        }
      };

      map.on('load', () => {
          map.addLayer({
          id: 'route',
          type: 'line',
          source: {
            type: 'geojson',
            data: geojson
          },
          layout: {
            'line-join': 'round',
            'line-cap': 'round'
          },
          paint: {
            'line-color': '#063E3F',
            'line-width': 5,
            'line-opacity': 0.85
          }
        });
        })

    },
    createMap: () => {
      // const mapboxgl = require('mapbox-gl/dist/mapbox-gl.js');
      mapboxgl.accessToken = 'pk.eyJ1Ijoic3RlZmFucmFmYWVsIiwiYSI6ImNsMWV2MG1vdTA2YXkzZG54dXBpZGJtd3IifQ.dFkQvD8fs51PMPrFOvaKKg';

      const map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v11',
        center: [-122.664983, 45.523823],
        zoom: 9
      });
      
    }
  },
  computed: {
    
  }
};
</script>

<style>


.home-enter-active,
.home-leave-active {
  transition: opacity 0.5s;
}
.home-enter,
.home-leave-active {
  opacity: 0;
}
</style>