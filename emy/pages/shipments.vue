<template>
  <div class="flex">
    <Header />
    <div class="w-full p-10 py-4">
      <div class="pb-4">
        <p class="font-bold text-first-900 text-xl">SHIPMENTS</p>
      </div>
      <hr />
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
              placeholder="Search for shipments"
            />
          </div>
        </div>

        <div class="inline-flex rounded-md p-5">
          <button
            href="#"
            class="
              py-2
              px-4
              h-10
              text-sm
              font-medium
              text-gray-700
              bg-white
              rounded-l-lg
              border border-gray-200
              hover:bg-gray-100
              focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700
              dark:bg-gray-700
              dark:border-gray-600
              dark:text-white
              dark:hover:text-white
              dark:hover:bg-gray-600
              dark:focus:ring-first-500
              dark:focus:text-white
            "
          >
            Previous
          </button>
          <button
            href="#"
            v-for="(page, index) in pages"
            :key="index"
            class="
              py-2
              px-4
              h-10
              text-sm
              font-medium
              text-gray-900
              bg-white
              border-t border-b border-r border-gray-200
              hover:bg-gray-100 hover:text-first-700
              focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700
              dark:bg-gray-700
              dark:border-gray-600
              dark:text-white
              dark:hover:text-white
              dark:hover:bg-gray-600
              dark:focus:ring-first-500
              dark:focus:text-white
            "
          >
            {{ page }}
          </button>
          <button
            href="#"
            class="
              py-2
              px-4
              h-10
              text-sm
              font-medium
              text-gray-900
              bg-white
              rounded-r-md
              border-t border-b border-r border-gray-200
              hover:bg-gray-100 hover:text-first-700
              focus:z-10 focus:ring-2 focus:ring-first-700 focus:text-first-700
              dark:bg-gray-700
              dark:border-gray-600
              dark:text-white
              dark:hover:text-white
              dark:hover:bg-gray-600
              dark:focus:ring-first-500
              dark:focus:text-white
            "
          >
            Next
          </button>
        </div>
      </div>
      
      <div class="grid grid-cols-2 gap-6">
        <div v-for="shipment, index in routePlan" :key="index" class="">
          <NuxtLink to='/shipmentdetails' class="block py-8 px-10 bg-white rounded-lg border border-gray-200 shadow-md hover:bg-gray-100 dark:bg-gray-800 dark:border-gray-700 dark:hover:bg-gray-700">
          <div class="flex justify-between">
            <div class="flex">
              <svg class="self-center h-10 w-10 mr-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M368 0C394.5 0 416 21.49 416 48V96H466.7C483.7 96 499.1 102.7 512 114.7L589.3 192C601.3 204 608 220.3 608 237.3V352C625.7 352 640 366.3 640 384C640 401.7 625.7 416 608 416H576C576 469 533 512 480 512C426.1 512 384 469 384 416H256C256 469 213 512 160 512C106.1 512 64 469 64 416H48C21.49 416 0 394.5 0 368V48C0 21.49 21.49 0 48 0H368zM416 160V256H544V237.3L466.7 160H416zM160 368C133.5 368 112 389.5 112 416C112 442.5 133.5 464 160 464C186.5 464 208 442.5 208 416C208 389.5 186.5 368 160 368zM480 464C506.5 464 528 442.5 528 416C528 389.5 506.5 368 480 368C453.5 368 432 389.5 432 416C432 442.5 453.5 464 480 464z"/></svg>
              <div class="">
                <p class="self-center text-lg font-bold tracking-tight text-gray-900 dark:text-white">Shipment Route {{index+1}}</p>
                <p class="text-md text-gray-500">Shipment Date</p>
              </div>
            </div>
            <div>
              <span class="self-center text-lg font-bold tracking-tight text-red-400 dark:text-white">+ 1,000</span><span class="self-center text-sm font-bold tracking-tight text-red-400 dark:text-white"> kgCO2eq</span>
              <p class="text-md text-gray-500">Emission</p>
            </div>
            <div class="flex">
              <svg class="self-center h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M256 64C256 46.33 270.3 32 288 32H415.1C415.1 32 415.1 32 415.1 32C420.3 32 424.5 32.86 428.2 34.43C431.1 35.98 435.5 38.27 438.6 41.3C438.6 41.35 438.6 41.4 438.7 41.44C444.9 47.66 447.1 55.78 448 63.9C448 63.94 448 63.97 448 64V192C448 209.7 433.7 224 416 224C398.3 224 384 209.7 384 192V141.3L214.6 310.6C202.1 323.1 181.9 323.1 169.4 310.6C156.9 298.1 156.9 277.9 169.4 265.4L338.7 96H288C270.3 96 256 81.67 256 64V64zM0 128C0 92.65 28.65 64 64 64H160C177.7 64 192 78.33 192 96C192 113.7 177.7 128 160 128H64V416H352V320C352 302.3 366.3 288 384 288C401.7 288 416 302.3 416 320V416C416 451.3 387.3 480 352 480H64C28.65 480 0 451.3 0 416V128z"/></svg>
            </div>
          </div>
          </NuxtLink>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  transition: "home",
  name: "Shipments",
  async asyncData({ $axios }) {
    const url = "http://localhost:8000";
    let response = await $axios.$get(`${url}/emy/routes`)
    let x = response

    var routePlan = []

    x.data.map((result) => {
      result.real_distance_traveled.map((route) => {
        routePlan.push(route)
      })
    })

    return { routePlan };
  },
  data() {
    return {
      shipments: [1,2,3,4,5,6,7,8,9,10],
      message: "",
      pages: [1,2,3,4,5]
    };
  },
};
</script>

<style>
  .home-enter-active, .home-leave-active { transition: opacity .5s; }
  .home-enter, .home-leave-active { opacity: 0; }
</style>