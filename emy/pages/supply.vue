<template>
  <div class="flex">
    <Header />
    <div class="p-10 py-4 w-full relative overflow-x-auto">
        <div class="pb-4">
          <p class="font-bold text-first-900 text-xl">SUPPLY DATA</p>
        </div>
      <hr />
      <div class="flex justify-between">
        <div class="flex">
          <div class="py-4">
            <label for="table-search" class="sr-only">Search</label>
            <div class="relative mt-1">
              <div
                class="
                  absolute
                  inset-y-0
                  left-0
                  flex
                  items-center
                  pl-3
                  pointer-events-none
                "
              >
                <svg
                  class="w-5 h-5 text-gray-500 dark:text-gray-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    fill-rule="evenodd"
                    d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                    clip-rule="evenodd"
                  ></path>
                </svg>
              </div>
              <input
                type="text"
                id="table-search"
                class="
                  bg-gray-50
                  border border-gray-300
                  text-gray-900 text-sm
                  rounded-lg
                  focus:ring-blue-500 focus:border-blue-500
                  block
                  w-80
                  pl-10
                  p-2
                  dark:bg-gray-700
                  dark:border-gray-600
                  dark:placeholder-gray-400
                  dark:text-white
                  dark:focus:ring-blue-500
                  dark:focus:border-blue-500
                "
                placeholder="Search for items"
              />
            </div>
          </div>
        
        <div class="inline-flex rounded-md p-5">
        <button href="#" class="py-2 px-4 h-10 text-sm font-medium text-gray-700 bg-white rounded-l-lg border border-gray-200 hover:bg-gray-100 focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:hover:text-white dark:hover:bg-gray-600 dark:focus:ring-first-500 dark:focus:text-white">
        Previous
        </button>
        <button href="#" v-for="page, index in pages" :key="index" class="py-2 px-4 h-10 text-sm font-medium text-gray-900 bg-white border-t border-b border-r border-gray-200 hover:bg-gray-100 hover:text-first-700 focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:hover:text-white dark:hover:bg-gray-600 dark:focus:ring-first-500 dark:focus:text-white">
        {{page}}
        </button>
        <button href="#" class="py-2 px-4 h-10 text-sm font-medium text-gray-900 bg-white rounded-r-md border-t border-b border-r border-gray-200 hover:bg-gray-100 hover:text-first-700 focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:hover:text-white dark:hover:bg-gray-600 dark:focus:ring-first-500 dark:focus:text-white">
        Next
        </button>
        </div>
        </div>

        <div class="py-5">
          <NuxtLink
            to="/routeplans"
            class="
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
            GET OPTIMIZED ROUTE
          </NuxtLink>
        </div>
      </div>
      <table class="w-full text-left text-gray-500 dark:text-gray-400">
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
            <th scope="col" class="px-6 py-3">Date</th>
            <th scope="col" class="px-6 py-3">Item</th>
            <th scope="col" class="px-6 py-3">Quantity</th>
            <th scope="col" class="px-6 py-3">Weight</th>
            <th scope="col" class="px-6 py-3">Total Weight</th>
            <th scope="col" class="px-6 py-3">Origin</th>
            <th scope="col" class="px-6 py-3">Destination</th>
            <th scope="col" class="px-6 py-3">
              <span class="sr-only">Edit</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="shipment, index in shipments" :key='index' :class="{'bg-gray-100': index % 2 === 1, '': index % 2 === 0}" class="text-sm bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
            <th scope="row" class="px-6 py-3">{{index+1}}</th>
            <td class="px-6 py-3">{{shipment.date_dropoff}}</td>
            <td class="px-6 py-3 font-medium text-gray-900 dark:text-white whitespace-nowrap">{{shipment.item_name}}</td>
            <td class="px-6 py-3">{{shipment.item_qty}}</td>
            <td class="px-6 py-3">{{shipment.item_weight}}</td>
            <td class="px-6 py-3">{{shipment.item_qty * shipment.item_weight}}</td>
            <td class="px-6 py-3">{{shipment.orig_store}}</td>
            <td class="px-6 py-3">{{shipment.dest_store}}</td>
            <td class="px-6 py-3">
              <div class="flex text-right justify-end">
              <button class="mx-2"><svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#174849" d="M490.3 40.4C512.2 62.27 512.2 97.73 490.3 119.6L460.3 149.7L362.3 51.72L392.4 21.66C414.3-.2135 449.7-.2135 471.6 21.66L490.3 40.4zM172.4 241.7L339.7 74.34L437.7 172.3L270.3 339.6C264.2 345.8 256.7 350.4 248.4 353.2L159.6 382.8C150.1 385.6 141.5 383.4 135 376.1C128.6 370.5 126.4 361 129.2 352.4L158.8 263.6C161.6 255.3 166.2 247.8 172.4 241.7V241.7zM192 63.1C209.7 63.1 224 78.33 224 95.1C224 113.7 209.7 127.1 192 127.1H96C78.33 127.1 64 142.3 64 159.1V416C64 433.7 78.33 448 96 448H352C369.7 448 384 433.7 384 416V319.1C384 302.3 398.3 287.1 416 287.1C433.7 287.1 448 302.3 448 319.1V416C448 469 405 512 352 512H96C42.98 512 0 469 0 416V159.1C0 106.1 42.98 63.1 96 63.1H192z"/></svg></button>
              <button><svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#BB1C29" d="M135.2 17.69C140.6 6.848 151.7 0 163.8 0H284.2C296.3 0 307.4 6.848 312.8 17.69L320 32H416C433.7 32 448 46.33 448 64C448 81.67 433.7 96 416 96H32C14.33 96 0 81.67 0 64C0 46.33 14.33 32 32 32H128L135.2 17.69zM394.8 466.1C393.2 492.3 372.3 512 346.9 512H101.1C75.75 512 54.77 492.3 53.19 466.1L31.1 128H416L394.8 466.1z"/></svg></button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    
  </div>
</template>

<script>
export default {
  transition: 'home',
  name: "IndexPage",
  async asyncData({ $axios }) {
    const url = "https://backend-emy.vercel.app";
    let response = await $axios.$get(`${url}/emy/supply`)
    let shipments = response.data
    return { shipments };
  },
  data() {
    return {
          pages: [1,2,3,4,5],
          loading: false,
        };
      },
    };
</script>

<style>
  .home-enter-active, .home-leave-active { transition: opacity .5s; }
  .home-enter, .home-leave-active { opacity: 0; }
</style>