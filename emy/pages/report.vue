<template>
  <div class="flex">
    <Header />
    <div class="w-full p-10 py-4">
      <div class="pb-4">
        <p class="font-bold text-first-900 text-xl">EMISSION REPORT</p>
      </div>
      <hr>
      <div class="flex justify-between pt-6">
        <div date-rangepicker class="flex items-center">
          <div class="relative">
            <input value="2022-05-04" type="date" name="start" class="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5  dark:bg-gray-700" placeholder="Select date start">
          </div>
          <span class="mx-4 text-gray-500">to</span>
          <div class="relative">
            <input value="2022-05-04" type="date" name="start" class="bg-gray-50 border border-gray-300 text-gray-900 sm:text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5  dark:bg-gray-700" placeholder="Select date start">
          </div>
          <button
            type="button"
            class="
              mx-4
              focus:outline-none
              text-white
              bg-first-700
              hover:bg-first-200
              focus:ring-4 focus:ring-green-300
              font-bold
              rounded-lg
              text-sm
              px-5
              py-2.5
              dark:bg-first-700
              dark:hover:bg-green-700
              dark:focus:ring-green-800
            "
          >
            Submit
          </button>
        <div>
        </div>
        </div>
        <div >
          <button
            @click="downloadCSV('04/05/2022')"
            type="button"
            class="
              mx-4
              focus:outline-none
              text-white
              bg-first-700
              hover:bg-first-200
              focus:ring-4 focus:ring-green-300
              font-bold
              rounded-lg
              text-sm
              px-5
              py-2.5
              dark:bg-first-700
              dark:hover:bg-green-700
              dark:focus:ring-green-800
            "
          >
            Download CSV
          </button>
          <button
            @click="downloadJSON('04/05/2022')"
            type="button"
            class="
              mx-4
              focus:outline-none
              text-white
              bg-first-700
              hover:bg-first-200
              focus:ring-4 focus:ring-green-300
              font-bold
              rounded-lg
              text-sm
              px-5
              py-2.5
              dark:bg-first-700
              dark:hover:bg-green-700
              dark:focus:ring-green-800
            "
          >
            Download JSON
          </button>
        </div>
      </div>
      <table class="w-full text-left text-gray-500 dark:text-gray-400 mt-4">
        <thead
          class="
            text-sm 
            text-white
            uppercase
            bg-first-700
            dark:bg-gray-700 dark:text-gray-400
          "
        >
          <tr>
            <th scope="col" class="px-6 py-3">No</th>
            <th scope="col" class="px-6 py-3">Date/Time</th>
            <th scope="col" class="px-6 py-3">Shipment ID</th>
            <th scope="col" class="px-6 py-3">Truck ID</th>
            <th scope="col" class="px-6 py-3">Start</th>
            <th scope="col" class="px-6 py-3">End</th>
            <th scope="col" class="px-6 py-3">CO2 Emission (kgCO2eq/t.km)</th>
            <th scope="col" class="px-6 py-3">Total Weight</th>
            <th scope="col" class="px-6 py-3">
              <span class="sr-only">Edit</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="shipment, name, index in shipments" :key='index' :class="{'bg-gray-100': index % 2 === 1, '': index % 2 === 0}" class="text-sm bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
            <th scope="row" class="px-6 py-3">{{index+1}}</th>
            <td class="px-6 py-3">{{'04/05/2022'}}</td>
            <td class="px-6 py-3 font-medium text-gray-900 dark:text-white whitespace-nowrap">{{response.data.shipmentId}}</td>
            <td class="px-6 py-3">{{name}}</td>
            <td class="px-6 py-3">{{shipment.Start}}</td>
            <td class="px-6 py-3">{{shipment.End}}</td>
            <td class="px-6 py-3">{{shipment.Emission}}</td>
            <td class="px-6 py-3">{{shipment.Weight}}</td>
            <td class="px-6 py-3">
              <div class="flex text-right justify-end">
              <button class="mx-2"><svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#174849" d="M490.3 40.4C512.2 62.27 512.2 97.73 490.3 119.6L460.3 149.7L362.3 51.72L392.4 21.66C414.3-.2135 449.7-.2135 471.6 21.66L490.3 40.4zM172.4 241.7L339.7 74.34L437.7 172.3L270.3 339.6C264.2 345.8 256.7 350.4 248.4 353.2L159.6 382.8C150.1 385.6 141.5 383.4 135 376.1C128.6 370.5 126.4 361 129.2 352.4L158.8 263.6C161.6 255.3 166.2 247.8 172.4 241.7V241.7zM192 63.1C209.7 63.1 224 78.33 224 95.1C224 113.7 209.7 127.1 192 127.1H96C78.33 127.1 64 142.3 64 159.1V416C64 433.7 78.33 448 96 448H352C369.7 448 384 433.7 384 416V319.1C384 302.3 398.3 287.1 416 287.1C433.7 287.1 448 302.3 448 319.1V416C448 469 405 512 352 512H96C42.98 512 0 469 0 416V159.1C0 106.1 42.98 63.1 96 63.1H192z"/></svg></button>
              <button ><svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#BB1C29" d="M135.2 17.69C140.6 6.848 151.7 0 163.8 0H284.2C296.3 0 307.4 6.848 312.8 17.69L320 32H416C433.7 32 448 46.33 448 64C448 81.67 433.7 96 416 96H32C14.33 96 0 81.67 0 64C0 46.33 14.33 32 32 32H128L135.2 17.69zM394.8 466.1C393.2 492.3 372.3 512 346.9 512H101.1C75.75 512 54.77 492.3 53.19 466.1L31.1 128H416L394.8 466.1z"/></svg></button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
      </div>
  </div>
</template>

<script>
import download from 'downloadjs'

export default {
  name: "Report",
  components: {
  },
  setup() {
  },
  async asyncData({ $axios }) {
    const url = "https://backend-emy.vercel.app";
    let response = await $axios.$get(`${url}/emy/get-shipment-data`)
    let shipments = response.data.truck
    return { response, shipments };
  },
  methods: {
    downloadCSV(date) {
     download(JSON.stringify(this.response.data.truck), `report-${date}.csv` , "text/plain")
    },
    downloadJSON(date) {
     download(JSON.stringify(this.response.data.truck), `report-${date}.json` , "text/plain")
    }
  },
  data() {
    return {
      shipments:[{
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          },
          {
          "_id":{
            "$oid":"6255efc1cce1054c0992a62e"
          },
          "date_dropoff":"4/5/2022",
          "order_id": 1,
          "item_id":"A54",
          "item_name":"ITEM54",
          "item_qty": 17,
          "item_weight": 4,
          "total_weight": 68,
          "orig_store":"Warehouse",
          "dest_store":"Shop_27"
          }],
      default: '04/05/2022'
        };
      },
  mounted() {
  }
    };
</script>