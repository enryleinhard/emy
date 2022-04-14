<!-- Please remove this file from your project -->
<template>
  <div class="h-screen">
    <aside class="w-64" aria-label="Sidebar">
      <div
        class="h-screen py-4 px-3 border-r"
      >
        <div>
          <a href="/" class="flex items-center mb-4">
            <img class="h-12" src="logo_ijo.png" alt="">
            <img class="h-8" src="emy.png" alt="">
          </a>
          <ul class="space-y-2">
            <li>
              <NuxtLink
                to="/supply"
                class="
                  flex
                  items-center
                  p-2
                  text-base
                  font-normal
                  text-gray-900
                  rounded-lg
                  dark:text-white
                  hover:bg-gray-100
                  dark:hover:bg-gray-700
                "
              >
              <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M448 32C483.3 32 512 60.65 512 96V416C512 451.3 483.3 480 448 480H64C28.65 480 0 451.3 0 416V96C0 60.65 28.65 32 64 32H448zM224 256V160H64V256H224zM64 320V416H224V320H64zM288 416H448V320H288V416zM448 256V160H288V256H448z"/></svg>
              <span class="flex-1 ml-3 whitespace-nowrap font-bold">Supply Data</span>
              </NuxtLink>
            </li>
            <li>
              <NuxtLink
                to="/routeplans"
                class="
                  flex
                  items-center
                  p-2
                  text-base
                  font-normal
                  text-gray-900
                  rounded-lg
                  dark:text-white
                  hover:bg-gray-100
                  dark:hover:bg-gray-700
                "
              >
              <svg class="w-6 h-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M320 256C302.3 256 288 270.3 288 288C288 305.7 302.3 320 320 320H416C469 320 512 362.1 512 416C512 469 469 512 416 512H139.6C148.3 502.1 158.9 489.4 169.6 475.2C175.9 466.8 182.4 457.6 188.6 448H416C433.7 448 448 433.7 448 416C448 398.3 433.7 384 416 384H320C266.1 384 223.1 341 223.1 288C223.1 234.1 266.1 192 320 192H362.1C340.2 161.5 320 125.4 320 96C320 42.98 362.1 0 416 0C469 0 512 42.98 512 96C512 160 416 256 416 256H320zM416 128C433.7 128 448 113.7 448 96C448 78.33 433.7 64 416 64C398.3 64 384 78.33 384 96C384 113.7 398.3 128 416 128zM118.3 487.8C118.1 488 117.9 488.2 117.7 488.4C113.4 493.4 109.5 497.7 106.3 501.2C105.9 501.6 105.5 502 105.2 502.4C99.5 508.5 96 512 96 512C96 512 0 416 0 352C0 298.1 42.98 255.1 96 255.1C149 255.1 192 298.1 192 352C192 381.4 171.8 417.5 149.9 448C138.1 463.2 127.7 476.9 118.3 487.8L118.3 487.8zM95.1 384C113.7 384 127.1 369.7 127.1 352C127.1 334.3 113.7 320 95.1 320C78.33 320 63.1 334.3 63.1 352C63.1 369.7 78.33 384 95.1 384z"/></svg>
              <span class="flex-1 ml-3 whitespace-nowrap font-bold">Route Plans</span>
              </NuxtLink>
            </li>
            <li>
              <NuxtLink
                to="/shipments"
                class="
                  flex
                  items-center
                  p-2
                  text-base
                  font-normal
                  text-gray-900
                  rounded-lg
                  dark:text-white
                  hover:bg-gray-100
                  dark:hover:bg-gray-700
                "
              >
              <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M368 0C394.5 0 416 21.49 416 48V96H466.7C483.7 96 499.1 102.7 512 114.7L589.3 192C601.3 204 608 220.3 608 237.3V352C625.7 352 640 366.3 640 384C640 401.7 625.7 416 608 416H576C576 469 533 512 480 512C426.1 512 384 469 384 416H256C256 469 213 512 160 512C106.1 512 64 469 64 416H48C21.49 416 0 394.5 0 368V48C0 21.49 21.49 0 48 0H368zM416 160V256H544V237.3L466.7 160H416zM160 368C133.5 368 112 389.5 112 416C112 442.5 133.5 464 160 464C186.5 464 208 442.5 208 416C208 389.5 186.5 368 160 368zM480 464C506.5 464 528 442.5 528 416C528 389.5 506.5 368 480 368C453.5 368 432 389.5 432 416C432 442.5 453.5 464 480 464z"/></svg>
              <span class="flex-1 ml-3 whitespace-nowrap font-bold">Past Shipments</span>
              </NuxtLink>
            </li>
            <li>
              <NuxtLink
                to="/report"
                class="
                  flex
                  items-center
                  p-2
                  text-base
                  font-normal
                  text-gray-900
                  rounded-lg
                  dark:text-white
                  hover:bg-gray-100
                  dark:hover:bg-gray-700
                "
              >
              <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Pro 6.1.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path fill="#063E3F" d="M448 336v-288C448 21.49 426.5 0 400 0H96C42.98 0 0 42.98 0 96v320c0 53.02 42.98 96 96 96h320c17.67 0 32-14.33 32-31.1c0-11.72-6.607-21.52-16-27.1v-81.36C441.8 362.8 448 350.2 448 336zM143.1 128h192C344.8 128 352 135.2 352 144C352 152.8 344.8 160 336 160H143.1C135.2 160 128 152.8 128 144C128 135.2 135.2 128 143.1 128zM143.1 192h192C344.8 192 352 199.2 352 208C352 216.8 344.8 224 336 224H143.1C135.2 224 128 216.8 128 208C128 199.2 135.2 192 143.1 192zM384 448H96c-17.67 0-32-14.33-32-32c0-17.67 14.33-32 32-32h288V448z"/></svg>
              <span class="flex-1 ml-3 whitespace-nowrap font-bold">Report</span>
              </NuxtLink>
            </li>
          </ul>
        </div>
        <div>
        </div>
      </div>
    </aside>
  </div>
</template>

<script>
export default {
  name: "Header",
};
</script>
