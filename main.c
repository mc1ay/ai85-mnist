/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// ai85-mnist
// Created using ai8xize.py --verbose --test-dir proj --prefix ai85-mnist --checkpoint-file proj/ai85-mnist-qat8-q.pth-quantized.tar --config-file networks/mnist-chw-ai85.yaml --device MAX78000 --compact-data --mexpress --softmax

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"
#include <mxc_device.h>
#include <gpio.h>
#include <uart.h>
#include <mxc_delay.h>
#include "ff.h"

/***** Definitions *****/

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define MAXLEN 4096

/***** Globals *****/
FATFS* fs;      //FFat Filesystem Object
FATFS fs_obj;
FIL imagesfile; //FFat File Object for images
FIL labelsfile; //FFat File Object for labels
FRESULT err;    //FFat Result (Struct)
FILINFO fno;    //FFat File Information Object
DIR dir;        //FFat Directory Object
TCHAR message[MAXLEN], directory[MAXLEN], cwd[MAXLEN], filename[MAXLEN], volume_label[24], volume = '0';
TCHAR* FF_ERRORS[20];
DWORD clusters_free = 0, sectors_free = 0, sectors_total = 0, volume_sn = 0;
UINT bytes_written = 0, bytes_read = 0, mounted = 0;
BYTE work[4096];
mxc_gpio_cfg_t SDPowerEnablePin = {MXC_GPIO1, MXC_GPIO_PIN_12, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIO};

volatile uint32_t cnn_time; // Stopwatch

// SDHC Functions
int mount()
{
    fs = &fs_obj;

    if ((err = f_mount(fs, "", 1)) != FR_OK) {          //Mount the default drive to fs now
        printf("Error opening SD card: %s\n", FF_ERRORS[err]);
        f_mount(NULL, "", 0);
    }
    else {
        printf("SD card mounted.\n");
        mounted = 1;
    }

    f_getcwd(cwd, sizeof(cwd));                         //Set the Current working directory

    return err;
}

int umount()
{
    if ((err = f_mount(NULL, "", 0)) != FR_OK) {        //Unmount the default drive from its mount point
        printf("Error unmounting volume: %s\n", FF_ERRORS[err]);
    }
    else {
        printf("SD card unmounted.\n");
        mounted = 0;
    }

    return err;
}

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 1-channel 28x28 data input (784 bytes / 196 32-bit words):
// CHW 28x28, channel 0
static const uint32_t input_0[] = SAMPLE_INPUT_0;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 196);
}

// Expected output of layer 4 for ai85-mnist given the sample input (known-answer test)
// Delete this function for production code
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t sample_output[] = SAMPLE_OUTPUT;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) return CNN_FAIL;
  }

  return CNN_OK;
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_shift_q17p14_q15((q31_t *) ml_data, CNN_NUM_OUTPUTS, 4, ml_softmax);
}

int main(void)
{
  FF_ERRORS[0] = "FR_OK";
  FF_ERRORS[1] = "FR_DISK_ERR";
  FF_ERRORS[2] = "FR_INT_ERR";
  FF_ERRORS[3] = "FR_NOT_READY";
  FF_ERRORS[4] = "FR_NO_FILE";
  FF_ERRORS[5] = "FR_NO_PATH";
  FF_ERRORS[6] = "FR_INVLAID_NAME";
  FF_ERRORS[7] = "FR_DENIED";
  FF_ERRORS[8] = "FR_EXIST";
  FF_ERRORS[9] = "FR_INVALID_OBJECT";
  FF_ERRORS[10] = "FR_WRITE_PROTECTED";
  FF_ERRORS[11] = "FR_INVALID_DRIVE";
  FF_ERRORS[12] = "FR_NOT_ENABLED";
  FF_ERRORS[13] = "FR_NO_FILESYSTEM";
  FF_ERRORS[14] = "FR_MKFS_ABORTED";
  FF_ERRORS[15] = "FR_TIMEOUT";
  FF_ERRORS[16] = "FR_LOCKED";
  FF_ERRORS[17] = "FR_NOT_ENOUGH_CORE";
  FF_ERRORS[18] = "FR_TOO_MANY_OPEN_FILES";
  FF_ERRORS[19] = "FR_INVALID_PARAMETER";

  int32_t data_count = 10000;  // process 10000 images

  int i;
  int digs, tens;
  int magic_number;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Read Test Data from SDHC
  printf("Mounting SDHC\n");
  mount();

  printf("Opening Test Images File: ");
  if ((err = f_open(&imagesfile, "t10k-images-idx3-ubyte", FA_READ)) != FR_OK) {
      printf("Error opening file: %s\n", FF_ERRORS[err]);
      f_mount(NULL, "", 0);
      return err;
  }
  else printf("OK\n");

  printf("Checking magic number: ");

  if ((err = f_read(&imagesfile, &message, 4, &bytes_read)) != FR_OK) {
	  printf("Error reading file: %s\n", FF_ERRORS[err]);
	  f_mount(NULL, "", 0);
	  return err;
  }
  // Print result
  else {
	  magic_number = 0;
	  magic_number += message[0]<<24;
	  magic_number += message[1]<<16;
	  magic_number += message[2]<<8;
	  magic_number += message[3]<<0;
	  printf("%d\n", magic_number);
  }
  // Advance to starting position
  printf("Seeking to first pixel: ");

  if ((err = f_lseek(&imagesfile, 16)) != FR_OK) {
	  printf("Error %s\n", FF_ERRORS[err]);
	  f_mount(NULL, "", 0);
	  return err;
  }
  else printf("OK\n");

  printf("Opening Test Labels File: ");
  if ((err = f_open(&labelsfile, "t10k-labels-idx1-ubyte", FA_READ)) != FR_OK) {
      printf("Error opening file: %s\n", FF_ERRORS[err]);
      f_mount(NULL, "", 0);
      return err;
  }
  else printf("OK\n");

  printf("Checking magic number: ");

  if ((err = f_read(&labelsfile, &message, 4, &bytes_read)) != FR_OK) {
	  printf("Error reading file: %s\n", FF_ERRORS[err]);
	  f_mount(NULL, "", 0);
	  return err;
  }
  // Print result
  else {
	  magic_number = 0;
	  magic_number += message[0]<<24;
	  magic_number += message[1]<<16;
	  magic_number += message[2]<<8;
	  magic_number += message[3]<<0;
	  printf("%d\n", magic_number);
  }

  // Advance to starting position
  printf("Seeking to first label: ");

  if ((err = f_lseek(&labelsfile, 8)) != FR_OK) {
	  printf("Error %s\n", FF_ERRORS[err]);
	  f_mount(NULL, "", 0);
	  return err;
  }
  else printf("OK\n");

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("Running inference on %d images...\n", data_count);

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine

  // Loop over all images

  load_input(); // Load data input
  cnn_start(); // Start CNN processing

  SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
  while (cnn_time == 0)
    __WFI(); // Wait for CNN

  if (check_output() != CNN_OK) fail();
  softmax_layer();

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral

  printf("Classification results:\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
  }

  return 0;
}

