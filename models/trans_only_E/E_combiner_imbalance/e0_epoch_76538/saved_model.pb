ы╠
ЛБ
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02v2.3.0-rc2-23-gb36436b0878бл
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
┌
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ћ
valueІBѕ BЂ
╩
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
Г
layer_metrics
layer_regularization_losses
trainable_variables
	variables
	regularization_losses
non_trainable_variables
 metrics

!layers
 
OM
VARIABLE_VALUEVariable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
Г
"layer_metrics
#layer_regularization_losses
trainable_variables
	variables
regularization_losses
$non_trainable_variables
%metrics

&layers
 
 
 
Г
'layer_metrics
(layer_regularization_losses
trainable_variables
	variables
regularization_losses
)non_trainable_variables
*metrics

+layers
 
 
 
Г
,layer_metrics
-layer_regularization_losses
trainable_variables
	variables
regularization_losses
.non_trainable_variables
/metrics

0layers
 
 
 
Г
1layer_metrics
2layer_regularization_losses
trainable_variables
	variables
regularization_losses
3non_trainable_variables
4metrics

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
:         
*
dtype0*
shape:         

z
serving_default_input_2Placeholder*'
_output_shapes
:         
*
dtype0*
shape:         

▄
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1031292
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Й
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_1031441
Ц
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
GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_1031454╬Х
к
Ѓ
"__inference__wrapped_model_1031152
input_1
input_22
.functional_1_mul_layer_readvariableop_resource
identityѕ╣
%functional_1/mul_layer/ReadVariableOpReadVariableOp.functional_1_mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02'
%functional_1/mul_layer/ReadVariableOpб
*functional_1/mul_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*functional_1/mul_layer/strided_slice/stackд
,functional_1/mul_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,functional_1/mul_layer/strided_slice/stack_1д
,functional_1/mul_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,functional_1/mul_layer/strided_slice/stack_2З
$functional_1/mul_layer/strided_sliceStridedSlice-functional_1/mul_layer/ReadVariableOp:value:03functional_1/mul_layer/strided_slice/stack:output:05functional_1/mul_layer/strided_slice/stack_1:output:05functional_1/mul_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$functional_1/mul_layer/strided_sliceЕ
functional_1/mul_layer/mulMul-functional_1/mul_layer/strided_slice:output:0input_1*
T0*'
_output_shapes
:         
2
functional_1/mul_layer/mulй
'functional_1/mul_layer/ReadVariableOp_1ReadVariableOp.functional_1_mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02)
'functional_1/mul_layer/ReadVariableOp_1д
,functional_1/mul_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,functional_1/mul_layer/strided_slice_1/stackф
.functional_1/mul_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_1/mul_layer/strided_slice_1/stack_1ф
.functional_1/mul_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_1/mul_layer/strided_slice_1/stack_2ђ
&functional_1/mul_layer/strided_slice_1StridedSlice/functional_1/mul_layer/ReadVariableOp_1:value:05functional_1/mul_layer/strided_slice_1/stack:output:07functional_1/mul_layer/strided_slice_1/stack_1:output:07functional_1/mul_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&functional_1/mul_layer/strided_slice_1»
functional_1/mul_layer/mul_1Mul/functional_1/mul_layer/strided_slice_1:output:0input_2*
T0*'
_output_shapes
:         
2
functional_1/mul_layer/mul_1х
functional_1/mul_layer/addAddV2functional_1/mul_layer/mul:z:0 functional_1/mul_layer/mul_1:z:0*
T0*'
_output_shapes
:         
2
functional_1/mul_layer/addї
functional_1/re_lu/ReluRelufunctional_1/mul_layer/add:z:0*
T0*'
_output_shapes
:         
2
functional_1/re_lu/Reluф
2functional_1/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/tf_op_layer_Sum/Sum/reduction_indicesЂ
 functional_1/tf_op_layer_Sum/SumSum%functional_1/re_lu/Relu:activations:0;functional_1/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:         *
	keep_dims(2"
 functional_1/tf_op_layer_Sum/SumЫ
(functional_1/tf_op_layer_RealDiv/RealDivRealDiv%functional_1/re_lu/Relu:activations:0)functional_1/tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:         
2*
(functional_1/tf_op_layer_RealDiv/RealDivђ
IdentityIdentity,functional_1/tf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
::P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
С
h
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_1031208

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesІ
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:         *
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
░
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1031386

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:         
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╔
ѓ
.__inference_functional_1_layer_call_fn_1031282
input_1
input_2
unknown
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_10312772
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
д
Ъ
I__inference_functional_1_layer_call_and_return_conditional_losses_1031338
inputs_0
inputs_1%
!mul_layer_readvariableop_resource
identityѕњ
mul_layer/ReadVariableOpReadVariableOp!mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02
mul_layer/ReadVariableOpѕ
mul_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
mul_layer/strided_slice/stackї
mul_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice/stack_1ї
mul_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice/stack_2д
mul_layer/strided_sliceStridedSlice mul_layer/ReadVariableOp:value:0&mul_layer/strided_slice/stack:output:0(mul_layer/strided_slice/stack_1:output:0(mul_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mul_layer/strided_sliceЃ
mul_layer/mulMul mul_layer/strided_slice:output:0inputs_0*
T0*'
_output_shapes
:         
2
mul_layer/mulќ
mul_layer/ReadVariableOp_1ReadVariableOp!mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02
mul_layer/ReadVariableOp_1ї
mul_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice_1/stackљ
!mul_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!mul_layer/strided_slice_1/stack_1љ
!mul_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!mul_layer/strided_slice_1/stack_2▓
mul_layer/strided_slice_1StridedSlice"mul_layer/ReadVariableOp_1:value:0(mul_layer/strided_slice_1/stack:output:0*mul_layer/strided_slice_1/stack_1:output:0*mul_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mul_layer/strided_slice_1Ѕ
mul_layer/mul_1Mul"mul_layer/strided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:         
2
mul_layer/mul_1Ђ
mul_layer/addAddV2mul_layer/mul:z:0mul_layer/mul_1:z:0*
T0*'
_output_shapes
:         
2
mul_layer/adde

re_lu/ReluRelumul_layer/add:z:0*
T0*'
_output_shapes
:         
2

re_lu/Reluљ
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices═
tf_op_layer_Sum/SumSumre_lu/Relu:activations:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:         *
	keep_dims(2
tf_op_layer_Sum/SumЙ
tf_op_layer_RealDiv/RealDivRealDivre_lu/Relu:activations:0tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:         
2
tf_op_layer_RealDiv/RealDivs
IdentityIdentitytf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
::Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
М
љ
F__inference_mul_layer_layer_call_and_return_conditional_losses_1031176

inputs
inputs_1
readvariableop_resource
identityѕt
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
strided_slice/stack_2Ж
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
:         
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
strided_slice_1/stack_2Ш
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
:         
2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         
2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
::O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
¤
ё
.__inference_functional_1_layer_call_fn_1031346
inputs_0
inputs_1
unknown
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_10312582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
д
Ъ
I__inference_functional_1_layer_call_and_return_conditional_losses_1031315
inputs_0
inputs_1%
!mul_layer_readvariableop_resource
identityѕњ
mul_layer/ReadVariableOpReadVariableOp!mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02
mul_layer/ReadVariableOpѕ
mul_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
mul_layer/strided_slice/stackї
mul_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice/stack_1ї
mul_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice/stack_2д
mul_layer/strided_sliceStridedSlice mul_layer/ReadVariableOp:value:0&mul_layer/strided_slice/stack:output:0(mul_layer/strided_slice/stack_1:output:0(mul_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mul_layer/strided_sliceЃ
mul_layer/mulMul mul_layer/strided_slice:output:0inputs_0*
T0*'
_output_shapes
:         
2
mul_layer/mulќ
mul_layer/ReadVariableOp_1ReadVariableOp!mul_layer_readvariableop_resource*
_output_shapes
:*
dtype02
mul_layer/ReadVariableOp_1ї
mul_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
mul_layer/strided_slice_1/stackљ
!mul_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!mul_layer/strided_slice_1/stack_1љ
!mul_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!mul_layer/strided_slice_1/stack_2▓
mul_layer/strided_slice_1StridedSlice"mul_layer/ReadVariableOp_1:value:0(mul_layer/strided_slice_1/stack:output:0*mul_layer/strided_slice_1/stack_1:output:0*mul_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mul_layer/strided_slice_1Ѕ
mul_layer/mul_1Mul"mul_layer/strided_slice_1:output:0inputs_1*
T0*'
_output_shapes
:         
2
mul_layer/mul_1Ђ
mul_layer/addAddV2mul_layer/mul:z:0mul_layer/mul_1:z:0*
T0*'
_output_shapes
:         
2
mul_layer/adde

re_lu/ReluRelumul_layer/add:z:0*
T0*'
_output_shapes
:         
2

re_lu/Reluљ
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices═
tf_op_layer_Sum/SumSumre_lu/Relu:activations:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:         *
	keep_dims(2
tf_op_layer_Sum/SumЙ
tf_op_layer_RealDiv/RealDivRealDivre_lu/Relu:activations:0tf_op_layer_Sum/Sum:output:0*
T0*
_cloned(*'
_output_shapes
:         
2
tf_op_layer_RealDiv/RealDivs
IdentityIdentitytf_op_layer_RealDiv/RealDiv:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
::Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
╔
ѓ
.__inference_functional_1_layer_call_fn_1031263
input_1
input_2
unknown
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_10312582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
М
|
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1031408
inputs_0
inputs_1
identityr
RealDivRealDivinputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:         
2	
RealDiv_
IdentityIdentityRealDiv:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         
:         :Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Ц
M
1__inference_tf_op_layer_Sum_layer_call_fn_1031402

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_10312082
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
▓
a
5__inference_tf_op_layer_RealDiv_layer_call_fn_1031414
inputs_0
inputs_1
identityя
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_10312222
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         
:         :Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Э
▒
I__inference_functional_1_layer_call_and_return_conditional_losses_1031277

inputs
inputs_1
mul_layer_1031270
identityѕб!mul_layer/StatefulPartitionedCallЋ
!mul_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1mul_layer_1031270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_mul_layer_layer_call_and_return_conditional_losses_10311762#
!mul_layer/StatefulPartitionedCallз
re_lu/PartitionedCallPartitionedCall*mul_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_10311942
re_lu/PartitionedCallЁ
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_10312082!
tf_op_layer_Sum/PartitionedCall╝
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_10312222%
#tf_op_layer_RealDiv/PartitionedCallц
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0"^mul_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:2F
!mul_layer/StatefulPartitionedCall!mul_layer/StatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
ў
y
%__inference_signature_wrapper_1031292
input_1
input_2
unknown
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__wrapped_model_10311522
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
░
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1031194

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:         
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Щ
▒
I__inference_functional_1_layer_call_and_return_conditional_losses_1031232
input_1
input_2
mul_layer_1031186
identityѕб!mul_layer/StatefulPartitionedCallЋ
!mul_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2mul_layer_1031186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_mul_layer_layer_call_and_return_conditional_losses_10311762#
!mul_layer/StatefulPartitionedCallз
re_lu/PartitionedCallPartitionedCall*mul_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_10311942
re_lu/PartitionedCallЁ
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_10312082!
tf_op_layer_Sum/PartitionedCall╝
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_10312222%
#tf_op_layer_RealDiv/PartitionedCallц
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0"^mul_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:2F
!mul_layer/StatefulPartitionedCall!mul_layer/StatefulPartitionedCall:P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
╔
Ђ
+__inference_mul_layer_layer_call_fn_1031381
inputs_0
inputs_1
unknown
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_mul_layer_layer_call_and_return_conditional_losses_10311762
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
С
h
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_1031397

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesІ
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:         *
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Щ
▒
I__inference_functional_1_layer_call_and_return_conditional_losses_1031243
input_1
input_2
mul_layer_1031236
identityѕб!mul_layer/StatefulPartitionedCallЋ
!mul_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2mul_layer_1031236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_mul_layer_layer_call_and_return_conditional_losses_10311762#
!mul_layer/StatefulPartitionedCallз
re_lu/PartitionedCallPartitionedCall*mul_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_10311942
re_lu/PartitionedCallЁ
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_10312082!
tf_op_layer_Sum/PartitionedCall╝
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_10312222%
#tf_op_layer_RealDiv/PartitionedCallц
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0"^mul_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:2F
!mul_layer/StatefulPartitionedCall!mul_layer/StatefulPartitionedCall:P L
'
_output_shapes
:         

!
_user_specified_name	input_1:PL
'
_output_shapes
:         

!
_user_specified_name	input_2
Э
▒
I__inference_functional_1_layer_call_and_return_conditional_losses_1031258

inputs
inputs_1
mul_layer_1031251
identityѕб!mul_layer/StatefulPartitionedCallЋ
!mul_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1mul_layer_1031251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_mul_layer_layer_call_and_return_conditional_losses_10311762#
!mul_layer/StatefulPartitionedCallз
re_lu/PartitionedCallPartitionedCall*mul_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_10311942
re_lu/PartitionedCallЁ
tf_op_layer_Sum/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_10312082!
tf_op_layer_Sum/PartitionedCall╝
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_10312222%
#tf_op_layer_RealDiv/PartitionedCallц
IdentityIdentity,tf_op_layer_RealDiv/PartitionedCall:output:0"^mul_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:2F
!mul_layer/StatefulPartitionedCall!mul_layer/StatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
█
њ
F__inference_mul_layer_layer_call_and_return_conditional_losses_1031373
inputs_0
inputs_1
readvariableop_resource
identityѕt
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
strided_slice/stack_2Ж
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
:         
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
strided_slice_1/stack_2Ш
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
:         
2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         
2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
::Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
╦
z
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1031222

inputs
inputs_1
identityp
RealDivRealDivinputsinputs_1*
T0*
_cloned(*'
_output_shapes
:         
2	
RealDiv_
IdentityIdentityRealDiv:z:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         
:         :O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
Љ
C
'__inference_re_lu_layer_call_fn_1031391

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_10311942
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Э

{
#__inference__traced_restore_1031454
file_prefix
assignvariableop_variable

identity_2ѕбAssignVariableOpО
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*d
value[BYB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesњ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesх
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

Identityў
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
Ц
ќ
 __inference__traced_save_1031441
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b1484dee5e9d4f09ad7b7abc9e56991f/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЛ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*d
value[BYB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesї
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesЯ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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
¤
ё
.__inference_functional_1_layer_call_fn_1031354
inputs_0
inputs_1
unknown
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_10312772
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:         
:         
:22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*з
serving_default▀
;
input_10
serving_default_input_1:0         

;
input_20
serving_default_input_2:0         
G
tf_op_layer_RealDiv0
StatefulPartitionedCall:0         
tensorflow/serving/predict:¤t
д
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
*7&call_and_return_all_conditional_losses
8_default_save_signature"ѓ
_tf_keras_networkТ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "mul_layer", "config": {"layer was saved without config": true}, "name": "mul_layer", "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["mul_layer", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["re_lu/Relu", "Sum"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["re_lu", 0, 0, {}], ["tf_op_layer_Sum", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf_op_layer_RealDiv", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
в"У
_tf_keras_input_layer╚{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
в"У
_tf_keras_input_layer╚{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
Ф
w
trainable_variables
	variables
regularization_losses
	keras_api
9__call__
*:&call_and_return_all_conditional_losses"Ћ
_tf_keras_layerч{"class_name": "mul_layer", "name": "mul_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
у
trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
љ
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"Ђ
_tf_keras_layerу{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
л
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"┴
_tf_keras_layerД{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_RealDiv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["re_lu/Relu", "Sum"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
layer_metrics
layer_regularization_losses
trainable_variables
	variables
	regularization_losses
non_trainable_variables
 metrics

!layers
6__call__
8_default_save_signature
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
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
Г
"layer_metrics
#layer_regularization_losses
trainable_variables
	variables
regularization_losses
$non_trainable_variables
%metrics

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
Г
'layer_metrics
(layer_regularization_losses
trainable_variables
	variables
regularization_losses
)non_trainable_variables
*metrics

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
Г
,layer_metrics
-layer_regularization_losses
trainable_variables
	variables
regularization_losses
.non_trainable_variables
/metrics

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
Г
1layer_metrics
2layer_regularization_losses
trainable_variables
	variables
regularization_losses
3non_trainable_variables
4metrics

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
є2Ѓ
.__inference_functional_1_layer_call_fn_1031282
.__inference_functional_1_layer_call_fn_1031354
.__inference_functional_1_layer_call_fn_1031263
.__inference_functional_1_layer_call_fn_1031346└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
I__inference_functional_1_layer_call_and_return_conditional_losses_1031232
I__inference_functional_1_layer_call_and_return_conditional_losses_1031338
I__inference_functional_1_layer_call_and_return_conditional_losses_1031243
I__inference_functional_1_layer_call_and_return_conditional_losses_1031315└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѕ2Ё
"__inference__wrapped_model_1031152я
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *NбK
IџF
!і
input_1         

!і
input_2         

Н2м
+__inference_mul_layer_layer_call_fn_1031381б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_mul_layer_layer_call_and_return_conditional_losses_1031373б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_re_lu_layer_call_fn_1031391б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_re_lu_layer_call_and_return_conditional_losses_1031386б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
█2п
1__inference_tf_op_layer_Sum_layer_call_fn_1031402б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_1031397б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
5__inference_tf_op_layer_RealDiv_layer_call_fn_1031414б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Щ2э
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1031408б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
;B9
%__inference_signature_wrapper_1031292input_1input_2¤
"__inference__wrapped_model_1031152еXбU
NбK
IџF
!і
input_1         

!і
input_2         

ф "IфF
D
tf_op_layer_RealDiv-і*
tf_op_layer_RealDiv         
┌
I__inference_functional_1_layer_call_and_return_conditional_losses_1031232ї`б]
VбS
IџF
!і
input_1         

!і
input_2         

p

 
ф "%б"
і
0         

џ ┌
I__inference_functional_1_layer_call_and_return_conditional_losses_1031243ї`б]
VбS
IџF
!і
input_1         

!і
input_2         

p 

 
ф "%б"
і
0         

џ ▄
I__inference_functional_1_layer_call_and_return_conditional_losses_1031315јbб_
XбU
KџH
"і
inputs/0         

"і
inputs/1         

p

 
ф "%б"
і
0         

џ ▄
I__inference_functional_1_layer_call_and_return_conditional_losses_1031338јbб_
XбU
KџH
"і
inputs/0         

"і
inputs/1         

p 

 
ф "%б"
і
0         

џ ▒
.__inference_functional_1_layer_call_fn_1031263`б]
VбS
IџF
!і
input_1         

!і
input_2         

p

 
ф "і         
▒
.__inference_functional_1_layer_call_fn_1031282`б]
VбS
IџF
!і
input_1         

!і
input_2         

p 

 
ф "і         
┤
.__inference_functional_1_layer_call_fn_1031346Ђbб_
XбU
KџH
"і
inputs/0         

"і
inputs/1         

p

 
ф "і         
┤
.__inference_functional_1_layer_call_fn_1031354Ђbб_
XбU
KџH
"і
inputs/0         

"і
inputs/1         

p 

 
ф "і         
Л
F__inference_mul_layer_layer_call_and_return_conditional_losses_1031373єZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         

ф "%б"
і
0         

џ е
+__inference_mul_layer_layer_call_fn_1031381yZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         

ф "і         
ъ
B__inference_re_lu_layer_call_and_return_conditional_losses_1031386X/б,
%б"
 і
inputs         

ф "%б"
і
0         

џ v
'__inference_re_lu_layer_call_fn_1031391K/б,
%б"
 і
inputs         

ф "і         
с
%__inference_signature_wrapper_1031292╣iбf
б 
_ф\
,
input_1!і
input_1         

,
input_2!і
input_2         
"IфF
D
tf_op_layer_RealDiv-і*
tf_op_layer_RealDiv         
п
P__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1031408ЃZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         
ф "%б"
і
0         

џ »
5__inference_tf_op_layer_RealDiv_layer_call_fn_1031414vZбW
PбM
KџH
"і
inputs/0         

"і
inputs/1         
ф "і         
е
L__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_1031397X/б,
%б"
 і
inputs         

ф "%б"
і
0         
џ ђ
1__inference_tf_op_layer_Sum_layer_call_fn_1031402K/б,
%б"
 і
inputs         

ф "і         