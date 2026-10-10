/*!
    \file    main.c
    \brief    
    
    \version  
*/

/*
    Copyright (c) 2023, GigaDevice Semiconductor Inc.

    Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this 
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, 
       this list of conditions and the following disclaimer in the documentation 
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors 
       may be used to endorse or promote products derived from this software without 
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <string.h>
#include "gd32h759i_eval.h"
#include "gd_nn_interface.h"
#include "nn_model_configure.h"
#include "nn_model_benchmark.h"
#include "yolo_postprocess.h"

void mpu_config(void);

/* input and output code */
int8_t* input_data = (int8_t*)static_buffer_peak;  
int8_t* output_data_ptr[2];  
    
/* nn_report init macro */
#define nn_report_init(nn_report, clock, m_name, opera_name) \
    nn_report.core_clock = clock;\
    nn_report.model_name = m_name;\
    nn_report.operator_name = (char**)opera_name;\

/* model reporter instance, if need invoke two models, please create two nn_report*/
nn_model_report_struct nn_report;

/* gd model_paras_array_dict arr, if need invoke two models, please create two buf*/
nn_uint8 model_paras_array_info_buf[8];

/* gd model_paras_array addr and model_paras_data addr */
const nn_uint8* model_paras_array_and_data[2] = {model_paras_arr, model_paras_data};

/* benchmark struct instance */
#if defined (BENCHMARK)
    static  nn_benchmark benchmark;                         
#endif

/* model parameters struct */
static nn_model plate_best_sim; 

#define nn_model_struct_init(m_struct)\
    m_struct.user_input = input_data;\
    m_struct.user_input_size = INPUT_SIZE * INPUT_TYPE_SIZE;\
    m_struct.user_output = (void**)output_data_ptr;\
    m_struct.user_output_size = OUTPUT_SIZE * OUTPUT_TYPE_SIZE;\
    \
    m_struct.operators_cb_array = func_cb_arr;\
    m_struct.model_paras_array = (const nn_uint8*)model_paras_array_and_data;\
    m_struct.model_paras_array_dict = model_paras_array_info_buf;\
    m_struct.report_ptr = &nn_report; 

/*!
    \brief      main function
    \param[in]  none
    \param[out] none
    \retval     none
*/
int main(void)
{

    /* error_record */
    error_type ret_error = ok;
#if defined(GD32H7XX) || defined(GD32H77D) || defined(GD32H77E)
    /* Enable I-Cache */
    SCB_EnableICache();
    /* Enable D-Cache */
    SCB_EnableDCache();
#endif
    /* systick */
    systick_config();
    /* mpu_config */
    mpu_config();
    
    /* model init */
    nn_model_struct_init(plate_best_sim);
    ret_error = nn_model_init(&plate_best_sim);
    if(ret_error != ok) {
        printf("init error: %d\n", ret_error);
        return -1;
    }

    /* report init */
    nn_report_init(nn_report, SystemCoreClock, model_name, operator_name);

    /* MCU Benchmark with PC */
    #if defined (BENCHMARK)

        benchmark_state state = benchmark_ok;
        
        /* init benchmark from up software */
        state = nn_benchmark_init(&benchmark); 
            
        /* run benchmark, receive data from PC */
        state = nn_benchmark_start(&benchmark, &plate_best_sim);

        if(state != benchmark_ok) {
            return -1;
        }

    /* regular invoke step */
    #else

        int i, j;
        /* uart init */
        gd_eval_com_init(EVAL_COM);

        printf("\r\n=== YOLO-FastestV2 Plate Detection ===\r\n");
        printf("Input: %dx%d, INT8, NHWC, scale=%.6f zp=%d\r\n", 192, 192, 0.003922f, -128);
        printf("Output0: 6x6x16 (stride=32)\r\n");
        printf("Output1: 12x12x16 (stride=16)\r\n");
        printf("3 anchors/scale, K-means anchors\r\n");

        for (i = 0; i < 10; i ++){
            
            /* fill input with test data (placeholder: all zeros = black image) */
            /* In real deployment, replace with camera frame preprocessing */
            memset(input_data, 0, INPUT_SIZE * INPUT_TYPE_SIZE);
            
            /* model invoke */
            ret_error = nn_model_invoke(&plate_best_sim);

            if(ret_error != ok) {
                printf("invoke error: %d\r\n", ret_error);
                continue;
            }

            /* Get output pointers */
            /* output_data_ptr[0] -> [1,6,6,16] (stride=32, 大anchor) — 由 nn_model 决定顺序 */
            /* output_data_ptr[1] -> [1,12,12,16] (stride=16, 小anchor) */

            /* YOLO post-processing */
            /* Quant params: scale=1/255=0.003922, zp=-128 (standard int8 symmetric) */
            detection_t detections[MAX_DETECTIONS];
            int num_dets = yolo_postprocess(
                output_data_ptr[0],     /* [1,6,6,16] int8 — stride=32 */
                output_data_ptr[1],     /* [1,12,12,16] int8 — stride=16 */
                0.0039215689f,          /* output_scale  = 1/255 */
                -128,                   /* output_zp     = -128 */
                detections,
                MAX_DETECTIONS
            );

            /* Print results */
            yolo_print_detections(detections, num_dets);

            /* printf report */
            gd_nn_report_printf(plate_best_sim.report_ptr);

        }

    #endif

    while(1) {
    
    }

}

/*!
    \brief      mpu config function
    \param[in]  none
    \param[out] none
    \retval     none
*/
void mpu_config(void)
{
    mpu_region_init_struct mpu_init_struct;
    mpu_region_struct_para_init(&mpu_init_struct);

    /* disable the MPU */
    ARM_MPU_Disable();
    ARM_MPU_SetRegion(0, 0);

    /* configure the MPU attributes for the entire 4GB area, Reserved, no access */
    /* This configuration is highly recommended to prevent Speculative Prefetching of external memory, 
       which may cause CPU read locks and even system errors */
    mpu_init_struct.region_base_address  = 0x0;
    mpu_init_struct.region_size          = MPU_REGION_SIZE_4GB;
    mpu_init_struct.access_permission    = MPU_AP_NO_ACCESS;
    mpu_init_struct.access_bufferable    = MPU_ACCESS_NON_BUFFERABLE;
    mpu_init_struct.access_cacheable     = MPU_ACCESS_NON_CACHEABLE;
    mpu_init_struct.access_shareable     = MPU_ACCESS_SHAREABLE;
    mpu_init_struct.region_number        = MPU_REGION_NUMBER0;
    mpu_init_struct.subregion_disable    = 0x87;
    mpu_init_struct.instruction_exec     = MPU_INSTRUCTION_EXEC_NOT_PERMIT;
    mpu_init_struct.tex_type             = MPU_TEX_TYPE0;
    mpu_region_config(&mpu_init_struct);
    mpu_region_enable();

    /* enable the MPU */
    ARM_MPU_Enable(MPU_MODE_PRIV_DEFAULT);
}

/* retarget the C library printf function to the USART */
#if defined(__GNUC__) && !defined(__clang__)/* For GNU GCC compiler */
/* retarget the C library printf function to the USART, in Eclipse GCC environment */
int __io_putchar(int ch)
{
    usart_data_transmit(EVAL_COM, (uint8_t) ch );
    while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));
    return ch;
}
#else
/* retarget the C library printf function to the USART */
int fputc(int ch, FILE *f)
{
    usart_data_transmit(EVAL_COM, (uint8_t)ch);
    while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));

    return ch;
}
#endif /* defined(__GNUC__) && !defined(__clang__) */