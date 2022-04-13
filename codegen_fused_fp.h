//This file is generated based on graph json file.
typedef struct FpTable
{
		char name[512];
		TVM_DLL int32_t (*fused_fp)(void*, void*, int32_t, void*, void*);
} FpTable_;
FpTable_ g_fpTable[1024];
int fused_fn_count= 0;
void *lookup_fp(const char* fn_name)
{
	for (int i=0; i < fused_fn_count; i++)
	{
		if (strcmp(g_fpTable[i].name, fn_name) ==0)
		{
			return (void *) g_fpTable[i].fused_fp;
		}
	}
	return nullptr;
}
void setup_fp()
{
		int i=0;
		//Setup fused function pointers
		strcpy(g_fpTable[i].name, "fused_cast_add_clip_cast");
		g_fpTable[i].fused_fp = fused_cast_add_clip_cast;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_7");
		g_fpTable[i].fused_fp = fused_nn_pad_7;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_9");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_9;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_8");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_8;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_8");
		g_fpTable[i].fused_fp = fused_clip_8;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d_5");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d_5;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_6");
		g_fpTable[i].fused_fp = fused_nn_pad_6;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_8");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_8;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_7");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_7;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_7");
		g_fpTable[i].fused_fp = fused_clip_7;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d_4");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_5");
		g_fpTable[i].fused_fp = fused_nn_pad_5;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_7");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_7;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_6");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_6;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_6");
		g_fpTable[i].fused_fp = fused_clip_6;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d_3");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d_3;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_4");
		g_fpTable[i].fused_fp = fused_nn_pad_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_6");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_6;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_5");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_5;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_5");
		g_fpTable[i].fused_fp = fused_clip_5;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d_2");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d_2;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_3");
		g_fpTable[i].fused_fp = fused_nn_pad_3;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_5");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_5;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_4");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_4");
		g_fpTable[i].fused_fp = fused_clip_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d_1");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d_1;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_2");
		g_fpTable[i].fused_fp = fused_nn_pad_2;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_4");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_3");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_3;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_3");
		g_fpTable[i].fused_fp = fused_clip_3;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_max_pool2d");
		g_fpTable[i].fused_fp = fused_nn_max_pool2d;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad_1");
		g_fpTable[i].fused_fp = fused_nn_pad_1;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_3");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_3;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_2");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_2;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_2");
		g_fpTable[i].fused_fp = fused_clip_2;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_2");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_2;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_1");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_1;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_1");
		g_fpTable[i].fused_fp = fused_clip_1;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_pad");
		g_fpTable[i].fused_fp = fused_nn_pad;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_1");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_1;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip");
		g_fpTable[i].fused_fp = fused_clip;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtract_add_cl_14356226997310000548_");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtract_add_cl_14356226997310000548_;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_12");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_12;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_9");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_9;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_image_resize");
		g_fpTable[i].fused_fp = fused_clip_image_resize;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_subtract_fixed_point_multiply_add_clip_cast");
		g_fpTable[i].fused_fp = fused_cast_subtract_fixed_point_multiply_add_clip_cast;
		i++;
		strcpy(g_fpTable[i].name, "fused_concatenate_nn_pad");
		g_fpTable[i].fused_fp = fused_concatenate_nn_pad;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_11");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_11;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_4");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_clip_4");
		g_fpTable[i].fused_fp = fused_clip_4;
		i++;
		strcpy(g_fpTable[i].name, "fused_nn_conv2d_subtract_add_10");
		g_fpTable[i].fused_fp = fused_nn_conv2d_subtract_add_10;
		i++;
		strcpy(g_fpTable[i].name, "fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtract_add_cl_14356226997310000548__1");
		g_fpTable[i].fused_fp = fused_cast_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtract_add_cl_14356226997310000548__1;
		i++;
		fused_fn_count = i;
}