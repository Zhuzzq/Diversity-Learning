ĹĎ
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878¤Ó
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
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
 
Y
w
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api

0

0
 
­
non_trainable_variables
	variables
layer_metrics
trainable_variables

layers
 metrics
!layer_regularization_losses
	regularization_losses
 
OM
VARIABLE_VALUEVariable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
"non_trainable_variables
	variables
#layer_metrics
trainable_variables

$layers
%metrics
&layer_regularization_losses
regularization_losses
 
 
 
­
'non_trainable_variables
	variables
(layer_metrics
trainable_variables

)layers
*metrics
+layer_regularization_losses
regularization_losses
 
 
 
­
,non_trainable_variables
	variables
-layer_metrics
trainable_variables

.layers
/metrics
0layer_regularization_losses
regularization_losses
 
 
 
­
1non_trainable_variables
	variables
2layer_metrics
trainable_variables

3layers
4metrics
5layer_regularization_losses
regularization_losses
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

Ú
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
GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_61822
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ź
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_61971
Ł
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_61984Öš

ł
G__inference_functional_1_layer_call_and_return_conditional_losses_61807

inputs
inputs_1
weight_layer_61800
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1weight_layer_61800*
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
GPU2*0J 8 *P
fKRI
G__inference_weight_layer_layer_call_and_return_conditional_losses_617062&
$weight_layer/StatefulPartitionedCallô
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
GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_617242
re_lu/PartitionedCall
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
GPU2*0J 8 *S
fNRL
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_617382!
tf_op_layer_Sum/PartitionedCallş
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
GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_617522%
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
Ë

,__inference_weight_layer_layer_call_fn_61911
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallú
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
GPU2*0J 8 *P
fKRI
G__inference_weight_layer_layer_call_and_return_conditional_losses_617062
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

ł
G__inference_functional_1_layer_call_and_return_conditional_losses_61762
input_1
input_2
weight_layer_61716
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2weight_layer_61716*
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
GPU2*0J 8 *P
fKRI
G__inference_weight_layer_layer_call_and_return_conditional_losses_617062&
$weight_layer/StatefulPartitionedCallô
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
GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_617242
re_lu/PartitionedCall
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
GPU2*0J 8 *S
fNRL
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_617382!
tf_op_layer_Sum/PartitionedCallş
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
GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_617522%
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

A
%__inference_re_lu_layer_call_fn_61921

inputs
identityÁ
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
GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_617242
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

 
_user_specified_nameinputs
É
x
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_61752

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
Ĺ

,__inference_functional_1_layer_call_fn_61793
input_1
input_2
unknown
identity˘StatefulPartitionedCallř
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
GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_617882
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
Ĺ

,__inference_functional_1_layer_call_fn_61812
input_1
input_2
unknown
identity˘StatefulPartitionedCallř
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
GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_618072
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
Ł

__inference__traced_save_61971
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
value3B1 B+_temp_909ec1f8e72b4051ac9157a4727dc15f/part2	
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

w
#__inference_signature_wrapper_61822
input_1
input_2
unknown
identity˘StatefulPartitionedCallŃ
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
GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_616822
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
Ń
z
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_61938
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
Ą
K
/__inference_tf_op_layer_Sum_layer_call_fn_61932

inputs
identityË
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
GPU2*0J 8 *S
fNRL
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_617382
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
Ü

G__inference_weight_layer_layer_call_and_return_conditional_losses_61903
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
Ô

G__inference_weight_layer_layer_call_and_return_conditional_losses_61706

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
Ë

,__inference_functional_1_layer_call_fn_61884
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallú
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
GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_618072
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
Ž
\
@__inference_re_lu_layer_call_and_return_conditional_losses_61916

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
Ž
\
@__inference_re_lu_layer_call_and_return_conditional_losses_61724

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
Ë

,__inference_functional_1_layer_call_fn_61876
inputs_0
inputs_1
unknown
identity˘StatefulPartitionedCallú
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
GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_617882
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
â
f
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_61927

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
â
f
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_61738

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
Â 

 __inference__wrapped_model_61682
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
˘
 
G__inference_functional_1_layer_call_and_return_conditional_losses_61845
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
Ž
_
3__inference_tf_op_layer_RealDiv_layer_call_fn_61944
inputs_0
inputs_1
identityÜ
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
GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_617522
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
ö

y
!__inference__traced_restore_61984
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

ł
G__inference_functional_1_layer_call_and_return_conditional_losses_61788

inputs
inputs_1
weight_layer_61781
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1weight_layer_61781*
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
GPU2*0J 8 *P
fKRI
G__inference_weight_layer_layer_call_and_return_conditional_losses_617062&
$weight_layer/StatefulPartitionedCallô
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
GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_617242
re_lu/PartitionedCall
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
GPU2*0J 8 *S
fNRL
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_617382!
tf_op_layer_Sum/PartitionedCallş
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
GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_617522%
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

ł
G__inference_functional_1_layer_call_and_return_conditional_losses_61773
input_1
input_2
weight_layer_61766
identity˘$weight_layer/StatefulPartitionedCall
$weight_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2weight_layer_61766*
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
GPU2*0J 8 *P
fKRI
G__inference_weight_layer_layer_call_and_return_conditional_losses_617062&
$weight_layer/StatefulPartitionedCallô
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
GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_617242
re_lu/PartitionedCall
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
GPU2*0J 8 *S
fNRL
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_617382!
tf_op_layer_Sum/PartitionedCallş
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
GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_617522%
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
˘
 
G__inference_functional_1_layer_call_and_return_conditional_losses_61868
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
inputs/1"¸L
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
tensorflow/serving/predict:Ąt
Ż
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
6__call__
*7&call_and_return_all_conditional_losses
8_default_save_signature"
_tf_keras_networkď{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "weight_layer", "config": {"layer was saved without config": true}, "name": "weight_layer", "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["weight_layer", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["re_lu/Relu", "Sum"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["re_lu", 0, 0, {}], ["tf_op_layer_Sum", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf_op_layer_RealDiv", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
ë"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ë"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
ą
w
	variables
trainable_variables
regularization_losses
	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "weight_layer", "name": "weight_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ç
	variables
trainable_variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"Ř
_tf_keras_layerž{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

	variables
trainable_variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layerç{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["re_lu/Relu", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
Đ
	variables
trainable_variables
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
non_trainable_variables
	variables
layer_metrics
trainable_variables

layers
 metrics
!layer_regularization_losses
	regularization_losses
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
­
"non_trainable_variables
	variables
#layer_metrics
trainable_variables

$layers
%metrics
&layer_regularization_losses
regularization_losses
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
'non_trainable_variables
	variables
(layer_metrics
trainable_variables

)layers
*metrics
+layer_regularization_losses
regularization_losses
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
,non_trainable_variables
	variables
-layer_metrics
trainable_variables

.layers
/metrics
0layer_regularization_losses
regularization_losses
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
1non_trainable_variables
	variables
2layer_metrics
trainable_variables

3layers
4metrics
5layer_regularization_losses
regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ţ2ű
,__inference_functional_1_layer_call_fn_61812
,__inference_functional_1_layer_call_fn_61793
,__inference_functional_1_layer_call_fn_61876
,__inference_functional_1_layer_call_fn_61884Ŕ
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
ę2ç
G__inference_functional_1_layer_call_and_return_conditional_losses_61868
G__inference_functional_1_layer_call_and_return_conditional_losses_61845
G__inference_functional_1_layer_call_and_return_conditional_losses_61762
G__inference_functional_1_layer_call_and_return_conditional_losses_61773Ŕ
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
2
 __inference__wrapped_model_61682Ţ
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
Ö2Ó
,__inference_weight_layer_layer_call_fn_61911˘
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
ń2î
G__inference_weight_layer_layer_call_and_return_conditional_losses_61903˘
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
Ď2Ě
%__inference_re_lu_layer_call_fn_61921˘
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
ę2ç
@__inference_re_lu_layer_call_and_return_conditional_losses_61916˘
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
Ů2Ö
/__inference_tf_op_layer_Sum_layer_call_fn_61932˘
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
ô2ń
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_61927˘
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
Ý2Ú
3__inference_tf_op_layer_RealDiv_layer_call_fn_61944˘
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
ř2ő
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_61938˘
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
9B7
#__inference_signature_wrapper_61822input_1input_2Í
 __inference__wrapped_model_61682¨X˘U
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
Ř
G__inference_functional_1_layer_call_and_return_conditional_losses_61762`˘]
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
 Ř
G__inference_functional_1_layer_call_and_return_conditional_losses_61773`˘]
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
 Ú
G__inference_functional_1_layer_call_and_return_conditional_losses_61845b˘_
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
 Ú
G__inference_functional_1_layer_call_and_return_conditional_losses_61868b˘_
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
 Ż
,__inference_functional_1_layer_call_fn_61793`˘]
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
Ż
,__inference_functional_1_layer_call_fn_61812`˘]
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
˛
,__inference_functional_1_layer_call_fn_61876b˘_
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
˛
,__inference_functional_1_layer_call_fn_61884b˘_
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

@__inference_re_lu_layer_call_and_return_conditional_losses_61916X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 t
%__inference_re_lu_layer_call_fn_61921K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "˙˙˙˙˙˙˙˙˙
á
#__inference_signature_wrapper_61822ši˘f
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
Ö
N__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_61938Z˘W
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
 ­
3__inference_tf_op_layer_RealDiv_layer_call_fn_61944vZ˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙

"
inputs/1˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
Ś
J__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_61927X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ~
/__inference_tf_op_layer_Sum_layer_call_fn_61932K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "˙˙˙˙˙˙˙˙˙Ň
G__inference_weight_layer_layer_call_and_return_conditional_losses_61903Z˘W
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
 Š
,__inference_weight_layer_layer_call_fn_61911yZ˘W
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