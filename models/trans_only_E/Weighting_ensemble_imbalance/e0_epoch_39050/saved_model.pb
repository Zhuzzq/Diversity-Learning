§Đ
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878âÓ
h
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0

NoOpNoOp
Ú
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
Ę
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
 
Y
w
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api

0

0
 
­
trainable_variables
layer_metrics
layer_regularization_losses
metrics
	variables
	regularization_losses
 non_trainable_variables

!layers
 
OM
VARIABLE_VALUEVariable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
trainable_variables
"layer_metrics
#layer_regularization_losses
$metrics
	variables
regularization_losses
%non_trainable_variables

&layers
 
 
 
­
trainable_variables
'layer_metrics
(layer_regularization_losses
)metrics
	variables
regularization_losses
*non_trainable_variables

+layers
 
 
 
­
trainable_variables
,layer_metrics
-layer_regularization_losses
.metrics
	variables
regularization_losses
/non_trainable_variables

0layers
 
 
 
­
trainable_variables
1layer_metrics
2layer_regularization_losses
3metrics
	variables
regularization_losses
4non_trainable_variables

5layers
 
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

z
serving_default_input_2Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

Ű
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_531368
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
˝
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_531517
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_531530ş
Í

-__inference_functional_1_layer_call_fn_531430
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5313532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1

x
$__inference_signature_wrapper_531368
input_1
input_2
unknown
identity˘StatefulPartitionedCallŇ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_5312282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2

ľ
H__inference_functional_1_layer_call_and_return_conditional_losses_531334

inputs
inputs_1
weight_layer_531327
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1weight_layer_531327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weight_layer_layer_call_and_return_conditional_losses_5312522&
$weight_layer/StatefulPartitionedCallő
re_lu/PartitionedCallPartitionedCall-weight_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_5312702
re_lu/PartitionedCall
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_5312842!
tf_op_layer_Sum/PartitionedCallť
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_5312982%
#tf_op_layer_RealDiv/PartitionedCall§
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0%^weight_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:2L
$weight_layer/StatefulPartitionedCall$weight_layer/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ł
L
0__inference_tf_op_layer_Sum_layer_call_fn_531478

inputs
identityĚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_5312842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ň
{
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_531484
inputs_0
inputs_1
identityr
RealDivRealDivinputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
RealDiv_
IdentityIdentityRealDiv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
Ł
Ą
H__inference_functional_1_layer_call_and_return_conditional_losses_531414
inputs_0
inputs_1(
$weight_layer_readvariableop_resource
identity
weight_layer/ReadVariableOpReadVariableOp$weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02
weight_layer/ReadVariableOp
 weight_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 weight_layer/strided_slice/stack
"weight_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice/stack_1
"weight_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice/stack_2¸
weight_layer/strided_sliceStridedSlice#weight_layer/ReadVariableOp:value:0)weight_layer/strided_slice/stack:output:0+weight_layer/strided_slice/stack_1:output:0+weight_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weight_layer/strided_slice
weight_layer/mulMul#weight_layer/strided_slice:output:0inputs_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/mul
weight_layer/ReadVariableOp_1ReadVariableOp$weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02
weight_layer/ReadVariableOp_1
"weight_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice_1/stack
$weight_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$weight_layer/strided_slice_1/stack_1
$weight_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$weight_layer/strided_slice_1/stack_2Ä
weight_layer/strided_slice_1StridedSlice%weight_layer/ReadVariableOp_1:value:0+weight_layer/strided_slice_1/stack:output:0-weight_layer/strided_slice_1/stack_1:output:0-weight_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weight_layer/strided_slice_1
weight_layer/mul_1Mul%weight_layer/strided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/mul_1
weight_layer/addAddV2weight_layer/mul:z:0weight_layer/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/addh

re_lu/ReluReluweight_layer/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

re_lu/Relu
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indicesÍ
tf_op_layer_Sum/SumSumre_lu/Relu:activations:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
tf_op_layer_Sum/Sumž
tf_op_layer_RealDiv/RealDivRealDivre_lu/Relu:activations:0tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
tf_op_layer_RealDiv/RealDivs
IdentityIdentitytf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
::Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1

ľ
H__inference_functional_1_layer_call_and_return_conditional_losses_531308
input_1
input_2
weight_layer_531262
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2weight_layer_531262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weight_layer_layer_call_and_return_conditional_losses_5312522&
$weight_layer/StatefulPartitionedCallő
re_lu/PartitionedCallPartitionedCall-weight_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_5312702
re_lu/PartitionedCall
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_5312842!
tf_op_layer_Sum/PartitionedCallť
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_5312982%
#tf_op_layer_RealDiv/PartitionedCall§
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0%^weight_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:2L
$weight_layer/StatefulPartitionedCall$weight_layer/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2
Ç

-__inference_functional_1_layer_call_fn_531339
input_1
input_2
unknown
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5313342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2
Ý

H__inference_weight_layer_layer_call_and_return_conditional_losses_531449
inputs_0
inputs_1
readvariableop_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ę
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
mulMulstrided_slice:output:0inputs_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
mulx
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ö
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
mul_1Mulstrided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
::Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1
Ő

H__inference_weight_layer_layer_call_and_return_conditional_losses_531252

inputs
inputs_1
readvariableop_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ę
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
mulMulstrided_slice:output:0inputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
mulx
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ö
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
mul_1Mulstrided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
÷

z
"__inference__traced_restore_531530
file_prefix
assignvariableop_variable

identity_2˘AssignVariableOp×
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*d
value[BYB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesľ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: :2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ż
]
A__inference_re_lu_layer_call_and_return_conditional_losses_531270

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Í

-__inference_functional_1_layer_call_fn_531422
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5313342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1

ľ
H__inference_functional_1_layer_call_and_return_conditional_losses_531353

inputs
inputs_1
weight_layer_531346
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1weight_layer_531346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weight_layer_layer_call_and_return_conditional_losses_5312522&
$weight_layer/StatefulPartitionedCallő
re_lu/PartitionedCallPartitionedCall-weight_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_5312702
re_lu/PartitionedCall
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_5312842!
tf_op_layer_Sum/PartitionedCallť
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_5312982%
#tf_op_layer_RealDiv/PartitionedCall§
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0%^weight_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:2L
$weight_layer/StatefulPartitionedCall$weight_layer/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ę
y
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_531298

inputs
inputs_1
identityp
RealDivRealDivinputsinputs_1*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
RealDiv_
IdentityIdentityRealDiv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ç

-__inference_functional_1_layer_call_fn_531358
input_1
input_2
unknown
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5313532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2
ă
g
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_531284

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ă
g
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_531473

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
°
`
4__inference_tf_op_layer_RealDiv_layer_call_fn_531490
inputs_0
inputs_1
identityÝ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_5312982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
Ż
]
A__inference_re_lu_layer_call_and_return_conditional_losses_531462

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ă 

!__inference__wrapped_model_531228
input_1
input_25
1functional_1_weight_layer_readvariableop_resource
identityÂ
(functional_1/weight_layer/ReadVariableOpReadVariableOp1functional_1_weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02*
(functional_1/weight_layer/ReadVariableOp¨
-functional_1/weight_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/weight_layer/strided_slice/stackŹ
/functional_1/weight_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/weight_layer/strided_slice/stack_1Ź
/functional_1/weight_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/weight_layer/strided_slice/stack_2
'functional_1/weight_layer/strided_sliceStridedSlice0functional_1/weight_layer/ReadVariableOp:value:06functional_1/weight_layer/strided_slice/stack:output:08functional_1/weight_layer/strided_slice/stack_1:output:08functional_1/weight_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/weight_layer/strided_slice˛
functional_1/weight_layer/mulMul0functional_1/weight_layer/strided_slice:output:0input_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
functional_1/weight_layer/mulĆ
*functional_1/weight_layer/ReadVariableOp_1ReadVariableOp1functional_1_weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/weight_layer/ReadVariableOp_1Ź
/functional_1/weight_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/weight_layer/strided_slice_1/stack°
1functional_1/weight_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1functional_1/weight_layer/strided_slice_1/stack_1°
1functional_1/weight_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1functional_1/weight_layer/strided_slice_1/stack_2
)functional_1/weight_layer/strided_slice_1StridedSlice2functional_1/weight_layer/ReadVariableOp_1:value:08functional_1/weight_layer/strided_slice_1/stack:output:0:functional_1/weight_layer/strided_slice_1/stack_1:output:0:functional_1/weight_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)functional_1/weight_layer/strided_slice_1¸
functional_1/weight_layer/mul_1Mul2functional_1/weight_layer/strided_slice_1:output:0input_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2!
functional_1/weight_layer/mul_1Á
functional_1/weight_layer/addAddV2!functional_1/weight_layer/mul:z:0#functional_1/weight_layer/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
functional_1/weight_layer/add
functional_1/re_lu/ReluRelu!functional_1/weight_layer/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
functional_1/re_lu/ReluŞ
2functional_1/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/tf_op_layer_Sum/Sum/reduction_indices
 functional_1/tf_op_layer_Sum/SumSum%functional_1/re_lu/Relu:activations:0;functional_1/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2"
 functional_1/tf_op_layer_Sum/Sumň
(functional_1/tf_op_layer_RealDiv/RealDivRealDiv%functional_1/re_lu/Relu:activations:0)functional_1/tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2*
(functional_1/tf_op_layer_RealDiv/RealDiv
IdentityIdentity,functional_1/tf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2
Ł
Ą
H__inference_functional_1_layer_call_and_return_conditional_losses_531391
inputs_0
inputs_1(
$weight_layer_readvariableop_resource
identity
weight_layer/ReadVariableOpReadVariableOp$weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02
weight_layer/ReadVariableOp
 weight_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 weight_layer/strided_slice/stack
"weight_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice/stack_1
"weight_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice/stack_2¸
weight_layer/strided_sliceStridedSlice#weight_layer/ReadVariableOp:value:0)weight_layer/strided_slice/stack:output:0+weight_layer/strided_slice/stack_1:output:0+weight_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weight_layer/strided_slice
weight_layer/mulMul#weight_layer/strided_slice:output:0inputs_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/mul
weight_layer/ReadVariableOp_1ReadVariableOp$weight_layer_readvariableop_resource*
_output_shapes
:*
dtype02
weight_layer/ReadVariableOp_1
"weight_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"weight_layer/strided_slice_1/stack
$weight_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$weight_layer/strided_slice_1/stack_1
$weight_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$weight_layer/strided_slice_1/stack_2Ä
weight_layer/strided_slice_1StridedSlice%weight_layer/ReadVariableOp_1:value:0+weight_layer/strided_slice_1/stack:output:0-weight_layer/strided_slice_1/stack_1:output:0-weight_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weight_layer/strided_slice_1
weight_layer/mul_1Mul%weight_layer/strided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/mul_1
weight_layer/addAddV2weight_layer/mul:z:0weight_layer/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
weight_layer/addh

re_lu/ReluReluweight_layer/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

re_lu/Relu
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indicesÍ
tf_op_layer_Sum/SumSumre_lu/Relu:activations:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
tf_op_layer_Sum/Sumž
tf_op_layer_RealDiv/RealDivRealDivre_lu/Relu:activations:0tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
tf_op_layer_RealDiv/RealDivs
IdentityIdentitytf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
::Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1

ľ
H__inference_functional_1_layer_call_and_return_conditional_losses_531319
input_1
input_2
weight_layer_531312
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2weight_layer_531312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weight_layer_layer_call_and_return_conditional_losses_5312522&
$weight_layer/StatefulPartitionedCallő
re_lu/PartitionedCallPartitionedCall-weight_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_5312702
re_lu/PartitionedCall
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_5312842!
tf_op_layer_Sum/PartitionedCallť
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_5312982%
#tf_op_layer_RealDiv/PartitionedCall§
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0%^weight_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:2L
$weight_layer/StatefulPartitionedCall$weight_layer/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_2
Í

-__inference_weight_layer_layer_call_fn_531457
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weight_layer_layer_call_and_return_conditional_losses_5312522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs/1
¤

__inference__traced_save_531517
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1044f8962f7d4ad3b01454fbc791ac6a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameŃ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*d
value[BYB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesŕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes

: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::

_output_shapes
: 

B
&__inference_re_lu_layer_call_fn_531467

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_5312702
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ó
serving_defaultß
;
input_10
serving_default_input_1:0˙˙˙˙˙˙˙˙˙

;
input_20
serving_default_input_2:0˙˙˙˙˙˙˙˙˙
G
tf_op_layer_RealDiv0
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict:Ĺt
Ż
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
6__call__
7_default_save_signature
*8&call_and_return_all_conditional_losses"
_tf_keras_networkď{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "weight_layer", "config": {"layer was saved without config": true}, "name": "weight_layer", "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["weight_layer", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["re_lu/Relu", "Sum"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["re_lu", 0, 0, {}], ["tf_op_layer_Sum", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf_op_layer_RealDiv", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
ë"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ë"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
ą
w
trainable_variables
	variables
regularization_losses
	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "weight_layer", "name": "weight_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ç
trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"Ř
_tf_keras_layerž{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layerç{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
Đ
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"Á
_tf_keras_layer§{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_RealDiv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["re_lu/Relu", "Sum"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
trainable_variables
layer_metrics
layer_regularization_losses
metrics
	variables
	regularization_losses
 non_trainable_variables

!layers
6__call__
7_default_save_signature
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
:2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
"layer_metrics
#layer_regularization_losses
$metrics
	variables
regularization_losses
%non_trainable_variables

&layers
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
'layer_metrics
(layer_regularization_losses
)metrics
	variables
regularization_losses
*non_trainable_variables

+layers
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
,layer_metrics
-layer_regularization_losses
.metrics
	variables
regularization_losses
/non_trainable_variables

0layers
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
1layer_metrics
2layer_regularization_losses
3metrics
	variables
regularization_losses
4non_trainable_variables

5layers
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2˙
-__inference_functional_1_layer_call_fn_531430
-__inference_functional_1_layer_call_fn_531358
-__inference_functional_1_layer_call_fn_531422
-__inference_functional_1_layer_call_fn_531339Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2
!__inference__wrapped_model_531228Ţ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *N˘K
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

î2ë
H__inference_functional_1_layer_call_and_return_conditional_losses_531414
H__inference_functional_1_layer_call_and_return_conditional_losses_531391
H__inference_functional_1_layer_call_and_return_conditional_losses_531319
H__inference_functional_1_layer_call_and_return_conditional_losses_531308Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
×2Ô
-__inference_weight_layer_layer_call_fn_531457˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_weight_layer_layer_call_and_return_conditional_losses_531449˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_re_lu_layer_call_fn_531467˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_re_lu_layer_call_and_return_conditional_losses_531462˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ú2×
0__inference_tf_op_layer_Sum_layer_call_fn_531478˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ő2ň
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_531473˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ţ2Ű
4__inference_tf_op_layer_RealDiv_layer_call_fn_531490˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ů2ö
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_531484˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
:B8
$__inference_signature_wrapper_531368input_1input_2Î
!__inference__wrapped_model_531228¨X˘U
N˘K
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

Ş "IŞF
D
tf_op_layer_RealDiv-*
tf_op_layer_RealDiv˙˙˙˙˙˙˙˙˙
Ů
H__inference_functional_1_layer_call_and_return_conditional_losses_531308`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 Ů
H__inference_functional_1_layer_call_and_return_conditional_losses_531319`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 Ű
H__inference_functional_1_layer_call_and_return_conditional_losses_531391b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 Ű
H__inference_functional_1_layer_call_and_return_conditional_losses_531414b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 °
-__inference_functional_1_layer_call_fn_531339`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

p

 
Ş "˙˙˙˙˙˙˙˙˙
°
-__inference_functional_1_layer_call_fn_531358`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙

!
input_2˙˙˙˙˙˙˙˙˙

p 

 
Ş "˙˙˙˙˙˙˙˙˙
ł
-__inference_functional_1_layer_call_fn_531422b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

p

 
Ş "˙˙˙˙˙˙˙˙˙
ł
-__inference_functional_1_layer_call_fn_531430b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

p 

 
Ş "˙˙˙˙˙˙˙˙˙

A__inference_re_lu_layer_call_and_return_conditional_losses_531462X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 u
&__inference_re_lu_layer_call_fn_531467K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "˙˙˙˙˙˙˙˙˙
â
$__inference_signature_wrapper_531368ši˘f
˘ 
_Ş\
,
input_1!
input_1˙˙˙˙˙˙˙˙˙

,
input_2!
input_2˙˙˙˙˙˙˙˙˙
"IŞF
D
tf_op_layer_RealDiv-*
tf_op_layer_RealDiv˙˙˙˙˙˙˙˙˙
×
O__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_531484Z˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 Ž
4__inference_tf_op_layer_RealDiv_layer_call_fn_531490vZ˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
§
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_531473X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
0__inference_tf_op_layer_Sum_layer_call_fn_531478K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "˙˙˙˙˙˙˙˙˙Ó
H__inference_weight_layer_layer_call_and_return_conditional_losses_531449Z˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 Ş
-__inference_weight_layer_layer_call_fn_531457yZ˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙

Ş "˙˙˙˙˙˙˙˙˙
