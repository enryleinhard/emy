<template>
  <div class="flex home">
    <Header />
    <div class="w-full p-10 py-4">
      <div class="pb-4">
        <p class="font-bold text-first-700 text-xl">ROUTE PLAN</p>
      </div>
      <hr />
<div class="w-full px-3 bg-white py-8">

	<div id="accordion-collapse" data-accordion="open">
		<h2 id="accordion-collapse-heading-1">
			<button type="button" class="flex items-center  justify-between p-5 w-full font-bold text-lgtext-left border border-b-0 text-first-900 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-t-xl" data-accordion-target="#accordion-collapse-body-1" aria-expanded="true" aria-controls="accordion-collapse-body-1">
      <span class="text-first-700 text-lg font-bold">{{ stringToDate(routePlan.date) }}</span>
      <svg data-accordion-icon class="w-6 h-6 shrink-0 rotate-180" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>
    </button>
		</h2>
		<div id="accordion-collapse-body-1" aria-labelledby="accordion-collapse-heading-1">
			<div class="p-5 border">
				<div v-for="shipment, index in shipments" :key="index" class="">
          <NuxtLink to='/plannedshipments' class="block py-8 px-10 bg-white rounded-lg border border-gray-200 shadow-md hover:bg-gray-100 dark:bg-gray-800 dark:border-gray-700 dark:hover:bg-gray-700">
          <div class="flex justify-between">
            <div class="flex">
              <svg class="self-center h-10 w-10 mr-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M368 0C394.5 0 416 21.49 416 48V96H466.7C483.7 96 499.1 102.7 512 114.7L589.3 192C601.3 204 608 220.3 608 237.3V352C625.7 352 640 366.3 640 384C640 401.7 625.7 416 608 416H576C576 469 533 512 480 512C426.1 512 384 469 384 416H256C256 469 213 512 160 512C106.1 512 64 469 64 416H48C21.49 416 0 394.5 0 368V48C0 21.49 21.49 0 48 0H368zM416 160V256H544V237.3L466.7 160H416zM160 368C133.5 368 112 389.5 112 416C112 442.5 133.5 464 160 464C186.5 464 208 442.5 208 416C208 389.5 186.5 368 160 368zM480 464C506.5 464 528 442.5 528 416C528 389.5 506.5 368 480 368C453.5 368 432 389.5 432 416C432 442.5 453.5 464 480 464z"/></svg>
              <div class="">
                <p class="self-center text-lg font-bold tracking-tight text-gray-900 dark:text-white">Shipment Route {{ index + 1 }}</p>
                <p class="text-md text-gray-500">{{ makeWaypoint(shipment) }}</p>
              </div>
            </div>
            <div class="flex">
              <svg class="self-center h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M256 64C256 46.33 270.3 32 288 32H415.1C415.1 32 415.1 32 415.1 32C420.3 32 424.5 32.86 428.2 34.43C431.1 35.98 435.5 38.27 438.6 41.3C438.6 41.35 438.6 41.4 438.7 41.44C444.9 47.66 447.1 55.78 448 63.9C448 63.94 448 63.97 448 64V192C448 209.7 433.7 224 416 224C398.3 224 384 209.7 384 192V141.3L214.6 310.6C202.1 323.1 181.9 323.1 169.4 310.6C156.9 298.1 156.9 277.9 169.4 265.4L338.7 96H288C270.3 96 256 81.67 256 64V64zM0 128C0 92.65 28.65 64 64 64H160C177.7 64 192 78.33 192 96C192 113.7 177.7 128 160 128H64V416H352V320C352 302.3 366.3 288 384 288C401.7 288 416 302.3 416 320V416C416 451.3 387.3 480 352 480H64C28.65 480 0 451.3 0 416V128z"/></svg>
            </div>
          </div>
          </NuxtLink>
        </div>
			</div>
		</div>
		<h2 id="accordion-collapse-heading-2">
			<button type="button" class="flex items-center focus:ring-4 focus:ring-first-700 dark:focus:ring-gray-800 justify-between p-5 w-full font-bold text-lgborder border-b-0 text-left text-first-900 hover:bg-gray-100 dark:hover:bg-gray-800" data-accordion-target="#accordion-collapse-body-2" aria-expanded="false" aria-controls="accordion-collapse-body-2">
      <span class="text-first-700 text-lg font-bold">3 May 2022</span>
      <svg data-accordion-icon class="w-6 h-6 shrink-0" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>
    </button>
		</h2>
		<div id="accordion-collapse-body-2" class="hidden" aria-labelledby="accordion-collapse-heading-2">
			<div class="p-5 border border-b-0">
				<p class="mb-2 text-first-900">No routes available.</p>
			</div>
		</div>
		<h2 id="accordion-collapse-heading-3">
			<button type="button" class="flex items-center border focus:ring-4 focus:ring-first-700 dark:focus:ring-gray-800 justify-between p-5 w-full font-bold text-lgtext-left text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800" data-accordion-target="#accordion-collapse-body-3" aria-expanded="false" aria-controls="accordion-collapse-body-3">
      <span class="text-first-700 text-lg font-bold">2 May 2022</span>
      <svg data-accordion-icon class="w-6 h-6 shrink-0" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>
    </button>
		</h2>
		<div id="accordion-collapse-body-3" class="hidden" aria-labelledby="accordion-collapse-heading-3">
			<div class="p-5 border border-t-0">
				<p class="mb-2 text-gray-500 dark:text-gray-400">No routes available.</p>
			</div>
		</div>
	</div>
	<script src="https://unpkg.com/flowbite@1.3.3/dist/flowbite.js"></script>
</div>
      <!-- <div class="px-10">
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
        <ol class="relative border-l left-2 top-2">
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
                bg-first-700
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
      </div> -->
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
  async asyncData({ $axios }) {
    const url = "https://backend-emy.vercel.app";
    let response = await $axios.$get(`${url}/emy/routes`)
    let routePlan = response.data[0]
    return { routePlan };
  },
  data() {
    return {
      accessToken: 'pk.eyJ1Ijoic3RlZmFucmFmYWVsIiwiYSI6ImNsMWV2MG1vdTA2YXkzZG54dXBpZGJtd3IifQ.dFkQvD8fs51PMPrFOvaKKg',
      shipments: [],
      message: "",
      pages: [1, 2, 3, 4, 5],
      routes: [],
      };
  },
  async mounted(){
  },
  created(){
    this.shipments = this.routePlan.routes;
    console.log(this.shipments)
  },
  
  methods: {
    makeWaypoint(route) {
      var string = ""
      for(let i = 0; i < route.length; i++){
        let x = route[i].shop_name;
        if(i == route.length - 1){
          string += x;
        }
        else{
          string += x + " -> ";
        }
      }

    

      console.log(string)
      return string
    },
    stringToDate(string) {
      var parts = string.split('/');
      var dates = new Date(parts[2], parts[1] - 1, parts[0]);

      dates = dates.toString().split(' ');
      return dates[2] + ' ' + dates[1] + ' ' + dates[3];
    },
    stringShipments() {
      var strShip = '';
      this.routes.map((r, index) => {
        if (index+1 == this.shipments.length) {
          strShip += `${r.shop_name}`
        } else {
          strShip += `${r.shop_name}  ->  `
        }
      })
      return strShip;
    },
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