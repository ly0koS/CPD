оч
ц§
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.1.02v2.1.0-15-g5466af38щ╝
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
ѓ
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
~
output_Zh/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*!
shared_nameoutput_Zh/kernel
w
$output_Zh/kernel/Read/ReadVariableOpReadVariableOpoutput_Zh/kernel* 
_output_shapes
:
ђђA*
dtype0
t
output_Zh/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_nameoutput_Zh/bias
m
"output_Zh/bias/Read/ReadVariableOpReadVariableOpoutput_Zh/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch1/kernel
y
%output_Ch1/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch1/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch1/bias
o
#output_Ch1/bias/Read/ReadVariableOpReadVariableOpoutput_Ch1/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch2/kernel
y
%output_Ch2/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch2/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch2/bias
o
#output_Ch2/bias/Read/ReadVariableOpReadVariableOpoutput_Ch2/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch3/kernel
y
%output_Ch3/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch3/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch3/bias
o
#output_Ch3/bias/Read/ReadVariableOpReadVariableOpoutput_Ch3/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch4/kernel
y
%output_Ch4/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch4/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch4/bias
o
#output_Ch4/bias/Read/ReadVariableOpReadVariableOpoutput_Ch4/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch5/kernel
y
%output_Ch5/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch5/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch5/bias
o
#output_Ch5/bias/Read/ReadVariableOpReadVariableOpoutput_Ch5/bias*
_output_shapes
:A*
dtype0
ђ
output_Ch6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*"
shared_nameoutput_Ch6/kernel
y
%output_Ch6/kernel/Read/ReadVariableOpReadVariableOpoutput_Ch6/kernel* 
_output_shapes
:
ђђA*
dtype0
v
output_Ch6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A* 
shared_nameoutput_Ch6/bias
o
#output_Ch6/bias/Read/ReadVariableOpReadVariableOpoutput_Ch6/bias*
_output_shapes
:A*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
ї
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
Ё
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
љ
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/m
Ѕ
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
ђ
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
љ
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m
Ѕ
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
ђ
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
ї
Adam/output_Zh/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*(
shared_nameAdam/output_Zh/kernel/m
Ё
+Adam/output_Zh/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Zh/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ѓ
Adam/output_Zh/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/output_Zh/bias/m
{
)Adam/output_Zh/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Zh/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch1/kernel/m
Є
,Adam/output_Ch1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch1/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch1/bias/m
}
*Adam/output_Ch1/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch1/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch2/kernel/m
Є
,Adam/output_Ch2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch2/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch2/bias/m
}
*Adam/output_Ch2/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch2/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch3/kernel/m
Є
,Adam/output_Ch3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch3/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch3/bias/m
}
*Adam/output_Ch3/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch3/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch4/kernel/m
Є
,Adam/output_Ch4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch4/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch4/bias/m
}
*Adam/output_Ch4/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch4/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch5/kernel/m
Є
,Adam/output_Ch5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch5/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch5/bias/m
}
*Adam/output_Ch5/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch5/bias/m*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch6/kernel/m
Є
,Adam/output_Ch6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch6/kernel/m* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch6/bias/m
}
*Adam/output_Ch6/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_Ch6/bias/m*
_output_shapes
:A*
dtype0
ї
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
Ё
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
љ
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/v
Ѕ
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
ђ
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
љ
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v
Ѕ
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
ђ
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
ї
Adam/output_Zh/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*(
shared_nameAdam/output_Zh/kernel/v
Ё
+Adam/output_Zh/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Zh/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ѓ
Adam/output_Zh/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/output_Zh/bias/v
{
)Adam/output_Zh/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Zh/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch1/kernel/v
Є
,Adam/output_Ch1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch1/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch1/bias/v
}
*Adam/output_Ch1/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch1/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch2/kernel/v
Є
,Adam/output_Ch2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch2/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch2/bias/v
}
*Adam/output_Ch2/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch2/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch3/kernel/v
Є
,Adam/output_Ch3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch3/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch3/bias/v
}
*Adam/output_Ch3/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch3/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch4/kernel/v
Є
,Adam/output_Ch4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch4/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch4/bias/v
}
*Adam/output_Ch4/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch4/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch5/kernel/v
Є
,Adam/output_Ch5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch5/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch5/bias/v
}
*Adam/output_Ch5/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch5/bias/v*
_output_shapes
:A*
dtype0
ј
Adam/output_Ch6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђA*)
shared_nameAdam/output_Ch6/kernel/v
Є
,Adam/output_Ch6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch6/kernel/v* 
_output_shapes
:
ђђA*
dtype0
ё
Adam/output_Ch6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*'
shared_nameAdam/output_Ch6/bias/v
}
*Adam/output_Ch6/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_Ch6/bias/v*
_output_shapes
:A*
dtype0

NoOpNoOp
╣Є
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*зє
valueУєBСє B▄є
┘
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
R
:	variables
;regularization_losses
<trainable_variables
=	keras_api
R
>	variables
?regularization_losses
@trainable_variables
A	keras_api
h

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
h

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
h

Zkernel
[bias
\	variables
]regularization_losses
^trainable_variables
_	keras_api
h

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
h

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
л
liter

mbeta_1

nbeta_2
	odecay
plearning_ratemЅmі&mІ'mї0mЇ1mјBmЈCmљHmЉImњNmЊOmћTmЋUmќZmЌ[mў`mЎamџfmЏgmюvЮvъ&vЪ'vа0vА1vбBvБCvцHvЦIvдNvДOvеTvЕUvфZvФ[vг`vГav«fv»gv░
ќ
0
1
&2
'3
04
15
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15
`16
a17
f18
g19
 
ќ
0
1
&2
'3
04
15
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15
`16
a17
f18
g19
џ

qlayers
	variables
regularization_losses
rnon_trainable_variables
slayer_regularization_losses
tmetrics
trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
џ

ulayers
	variables
regularization_losses
trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
 
 
 
џ

ylayers
	variables
regularization_losses
 trainable_variables
znon_trainable_variables
{layer_regularization_losses
|metrics
 
 
 
Џ

}layers
"	variables
#regularization_losses
$trainable_variables
~non_trainable_variables
layer_regularization_losses
ђmetrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
ъ
Ђlayers
(	variables
)regularization_losses
*trainable_variables
ѓnon_trainable_variables
 Ѓlayer_regularization_losses
ёmetrics
 
 
 
ъ
Ёlayers
,	variables
-regularization_losses
.trainable_variables
єnon_trainable_variables
 Єlayer_regularization_losses
ѕmetrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
ъ
Ѕlayers
2	variables
3regularization_losses
4trainable_variables
іnon_trainable_variables
 Іlayer_regularization_losses
їmetrics
 
 
 
ъ
Їlayers
6	variables
7regularization_losses
8trainable_variables
јnon_trainable_variables
 Јlayer_regularization_losses
љmetrics
 
 
 
ъ
Љlayers
:	variables
;regularization_losses
<trainable_variables
њnon_trainable_variables
 Њlayer_regularization_losses
ћmetrics
 
 
 
ъ
Ћlayers
>	variables
?regularization_losses
@trainable_variables
ќnon_trainable_variables
 Ќlayer_regularization_losses
ўmetrics
\Z
VARIABLE_VALUEoutput_Zh/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEoutput_Zh/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
ъ
Ўlayers
D	variables
Eregularization_losses
Ftrainable_variables
џnon_trainable_variables
 Џlayer_regularization_losses
юmetrics
][
VARIABLE_VALUEoutput_Ch1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
ъ
Юlayers
J	variables
Kregularization_losses
Ltrainable_variables
ъnon_trainable_variables
 Ъlayer_regularization_losses
аmetrics
][
VARIABLE_VALUEoutput_Ch2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
ъ
Аlayers
P	variables
Qregularization_losses
Rtrainable_variables
бnon_trainable_variables
 Бlayer_regularization_losses
цmetrics
][
VARIABLE_VALUEoutput_Ch3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
ъ
Цlayers
V	variables
Wregularization_losses
Xtrainable_variables
дnon_trainable_variables
 Дlayer_regularization_losses
еmetrics
][
VARIABLE_VALUEoutput_Ch4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
 

Z0
[1
ъ
Еlayers
\	variables
]regularization_losses
^trainable_variables
фnon_trainable_variables
 Фlayer_regularization_losses
гmetrics
][
VARIABLE_VALUEoutput_Ch5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
ъ
Гlayers
b	variables
cregularization_losses
dtrainable_variables
«non_trainable_variables
 »layer_regularization_losses
░metrics
][
VARIABLE_VALUEoutput_Ch6/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_Ch6/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
ъ
▒layers
h	variables
iregularization_losses
jtrainable_variables
▓non_trainable_variables
 │layer_regularization_losses
┤metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
8
х0
Х1
и2
И3
╣4
║5
╗6
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


╝total

йcount
Й
_fn_kwargs
┐	variables
└regularization_losses
┴trainable_variables
┬	keras_api


├total

─count
┼
_fn_kwargs
к	variables
Кregularization_losses
╚trainable_variables
╔	keras_api


╩total

╦count
╠
_fn_kwargs
═	variables
╬regularization_losses
¤trainable_variables
л	keras_api


Лtotal

мcount
М
_fn_kwargs
н	variables
Нregularization_losses
оtrainable_variables
О	keras_api


пtotal

┘count
┌
_fn_kwargs
█	variables
▄regularization_losses
Пtrainable_variables
я	keras_api


▀total

Яcount
р
_fn_kwargs
Р	variables
сregularization_losses
Сtrainable_variables
т	keras_api


Тtotal

уcount
У
_fn_kwargs
ж	variables
Жregularization_losses
вtrainable_variables
В	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

╝0
й1
 
 
А
ьlayers
┐	variables
└regularization_losses
┴trainable_variables
Ьnon_trainable_variables
 №layer_regularization_losses
­metrics
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

├0
─1
 
 
А
ыlayers
к	variables
Кregularization_losses
╚trainable_variables
Ыnon_trainable_variables
 зlayer_regularization_losses
Зmetrics
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

╩0
╦1
 
 
А
шlayers
═	variables
╬regularization_losses
¤trainable_variables
Шnon_trainable_variables
 эlayer_regularization_losses
Эmetrics
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

Л0
м1
 
 
А
щlayers
н	variables
Нregularization_losses
оtrainable_variables
Щnon_trainable_variables
 чlayer_regularization_losses
Чmetrics
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

п0
┘1
 
 
А
§layers
█	variables
▄regularization_losses
Пtrainable_variables
■non_trainable_variables
  layer_regularization_losses
ђmetrics
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

▀0
Я1
 
 
А
Ђlayers
Р	variables
сregularization_losses
Сtrainable_variables
ѓnon_trainable_variables
 Ѓlayer_regularization_losses
ёmetrics
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

Т0
у1
 
 
А
Ёlayers
ж	variables
Жregularization_losses
вtrainable_variables
єnon_trainable_variables
 Єlayer_regularization_losses
ѕmetrics
 

╝0
й1
 
 
 

├0
─1
 
 
 

╩0
╦1
 
 
 

Л0
м1
 
 
 

п0
┘1
 
 
 

▀0
Я1
 
 
 

Т0
у1
 
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_Zh/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/output_Zh/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch4/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch6/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch6/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_Zh/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/output_Zh/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch4/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/output_Ch6/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_Ch6/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ї
serving_default_inputPlaceholder*1
_output_shapes
:         ђђ*
dtype0*&
shape:         ђђ
ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasoutput_Ch6/kerneloutput_Ch6/biasoutput_Ch5/kerneloutput_Ch5/biasoutput_Ch4/kerneloutput_Ch4/biasoutput_Ch3/kerneloutput_Ch3/biasoutput_Ch2/kerneloutput_Ch2/biasoutput_Ch1/kerneloutput_Ch1/biasoutput_Zh/kerneloutput_Zh/bias* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_160992
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▄
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp$output_Zh/kernel/Read/ReadVariableOp"output_Zh/bias/Read/ReadVariableOp%output_Ch1/kernel/Read/ReadVariableOp#output_Ch1/bias/Read/ReadVariableOp%output_Ch2/kernel/Read/ReadVariableOp#output_Ch2/bias/Read/ReadVariableOp%output_Ch3/kernel/Read/ReadVariableOp#output_Ch3/bias/Read/ReadVariableOp%output_Ch4/kernel/Read/ReadVariableOp#output_Ch4/bias/Read/ReadVariableOp%output_Ch5/kernel/Read/ReadVariableOp#output_Ch5/bias/Read/ReadVariableOp%output_Ch6/kernel/Read/ReadVariableOp#output_Ch6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp+Adam/output_Zh/kernel/m/Read/ReadVariableOp)Adam/output_Zh/bias/m/Read/ReadVariableOp,Adam/output_Ch1/kernel/m/Read/ReadVariableOp*Adam/output_Ch1/bias/m/Read/ReadVariableOp,Adam/output_Ch2/kernel/m/Read/ReadVariableOp*Adam/output_Ch2/bias/m/Read/ReadVariableOp,Adam/output_Ch3/kernel/m/Read/ReadVariableOp*Adam/output_Ch3/bias/m/Read/ReadVariableOp,Adam/output_Ch4/kernel/m/Read/ReadVariableOp*Adam/output_Ch4/bias/m/Read/ReadVariableOp,Adam/output_Ch5/kernel/m/Read/ReadVariableOp*Adam/output_Ch5/bias/m/Read/ReadVariableOp,Adam/output_Ch6/kernel/m/Read/ReadVariableOp*Adam/output_Ch6/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp+Adam/output_Zh/kernel/v/Read/ReadVariableOp)Adam/output_Zh/bias/v/Read/ReadVariableOp,Adam/output_Ch1/kernel/v/Read/ReadVariableOp*Adam/output_Ch1/bias/v/Read/ReadVariableOp,Adam/output_Ch2/kernel/v/Read/ReadVariableOp*Adam/output_Ch2/bias/v/Read/ReadVariableOp,Adam/output_Ch3/kernel/v/Read/ReadVariableOp*Adam/output_Ch3/bias/v/Read/ReadVariableOp,Adam/output_Ch4/kernel/v/Read/ReadVariableOp*Adam/output_Ch4/bias/v/Read/ReadVariableOp,Adam/output_Ch5/kernel/v/Read/ReadVariableOp*Adam/output_Ch5/bias/v/Read/ReadVariableOp,Adam/output_Ch6/kernel/v/Read/ReadVariableOp*Adam/output_Ch6/bias/v/Read/ReadVariableOpConst*\
TinU
S2Q	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_161744
Ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasoutput_Zh/kerneloutput_Zh/biasoutput_Ch1/kerneloutput_Ch1/biasoutput_Ch2/kerneloutput_Ch2/biasoutput_Ch3/kerneloutput_Ch3/biasoutput_Ch4/kerneloutput_Ch4/biasoutput_Ch5/kerneloutput_Ch5/biasoutput_Ch6/kerneloutput_Ch6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/output_Zh/kernel/mAdam/output_Zh/bias/mAdam/output_Ch1/kernel/mAdam/output_Ch1/bias/mAdam/output_Ch2/kernel/mAdam/output_Ch2/bias/mAdam/output_Ch3/kernel/mAdam/output_Ch3/bias/mAdam/output_Ch4/kernel/mAdam/output_Ch4/bias/mAdam/output_Ch5/kernel/mAdam/output_Ch5/bias/mAdam/output_Ch6/kernel/mAdam/output_Ch6/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/output_Zh/kernel/vAdam/output_Zh/bias/vAdam/output_Ch1/kernel/vAdam/output_Ch1/bias/vAdam/output_Ch2/kernel/vAdam/output_Ch2/bias/vAdam/output_Ch3/kernel/vAdam/output_Ch3/bias/vAdam/output_Ch4/kernel/vAdam/output_Ch4/bias/vAdam/output_Ch5/kernel/vAdam/output_Ch5/bias/vAdam/output_Ch6/kernel/vAdam/output_Ch6/bias/v*[
TinT
R2P*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_161993ТВ
├
ф
)__inference_conv2d_2_layer_call_fn_160447

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1604392
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ќд
у
!__inference__wrapped_model_160360	
input/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource3
/model_output_ch6_matmul_readvariableop_resource4
0model_output_ch6_biasadd_readvariableop_resource3
/model_output_ch5_matmul_readvariableop_resource4
0model_output_ch5_biasadd_readvariableop_resource3
/model_output_ch4_matmul_readvariableop_resource4
0model_output_ch4_biasadd_readvariableop_resource3
/model_output_ch3_matmul_readvariableop_resource4
0model_output_ch3_biasadd_readvariableop_resource3
/model_output_ch2_matmul_readvariableop_resource4
0model_output_ch2_biasadd_readvariableop_resource3
/model_output_ch1_matmul_readvariableop_resource4
0model_output_ch1_biasadd_readvariableop_resource2
.model_output_zh_matmul_readvariableop_resource3
/model_output_zh_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕб#model/conv2d/BiasAdd/ReadVariableOpб"model/conv2d/Conv2D/ReadVariableOpб%model/conv2d_1/BiasAdd/ReadVariableOpб$model/conv2d_1/Conv2D/ReadVariableOpб%model/conv2d_2/BiasAdd/ReadVariableOpб$model/conv2d_2/Conv2D/ReadVariableOpб'model/output_Ch1/BiasAdd/ReadVariableOpб&model/output_Ch1/MatMul/ReadVariableOpб'model/output_Ch2/BiasAdd/ReadVariableOpб&model/output_Ch2/MatMul/ReadVariableOpб'model/output_Ch3/BiasAdd/ReadVariableOpб&model/output_Ch3/MatMul/ReadVariableOpб'model/output_Ch4/BiasAdd/ReadVariableOpб&model/output_Ch4/MatMul/ReadVariableOpб'model/output_Ch5/BiasAdd/ReadVariableOpб&model/output_Ch5/MatMul/ReadVariableOpб'model/output_Ch6/BiasAdd/ReadVariableOpб&model/output_Ch6/MatMul/ReadVariableOpб&model/output_Zh/BiasAdd/ReadVariableOpб%model/output_Zh/MatMul/ReadVariableOp╝
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp╦
model/conv2d/Conv2DConv2Dinput*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
model/conv2d/Conv2D│
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOpЙ
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
model/conv2d/BiasAddЅ
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
model/conv2d/ReluМ
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPoolю
model/dropout/IdentityIdentity$model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @@2
model/dropout/Identity┬
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpж
model/conv2d_1/Conv2DConv2Dmodel/dropout/Identity:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
model/conv2d_1/Conv2D╣
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp─
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
model/conv2d_1/BiasAddЇ
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
model/conv2d_1/Relu┘
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool┬
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp­
model/conv2d_2/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
model/conv2d_2/Conv2D╣
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp─
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
model/conv2d_2/BiasAddЇ
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
model/conv2d_2/Relu┘
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPoolб
model/dropout_1/IdentityIdentity&model/max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2
model/dropout_1/Identity{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     @  2
model/flatten/Const«
model/flatten/ReshapeReshape!model/dropout_1/Identity:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:         ђђ2
model/flatten/Reshape┬
&model/output_Ch6/MatMul/ReadVariableOpReadVariableOp/model_output_ch6_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch6/MatMul/ReadVariableOpЙ
model/output_Ch6/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch6/MatMul┐
'model/output_Ch6/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch6_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch6/BiasAdd/ReadVariableOp┼
model/output_Ch6/BiasAddBiasAdd!model/output_Ch6/MatMul:product:0/model/output_Ch6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch6/BiasAddћ
model/output_Ch6/SoftmaxSoftmax!model/output_Ch6/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch6/Softmax┬
&model/output_Ch5/MatMul/ReadVariableOpReadVariableOp/model_output_ch5_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch5/MatMul/ReadVariableOpЙ
model/output_Ch5/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch5/MatMul┐
'model/output_Ch5/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch5_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch5/BiasAdd/ReadVariableOp┼
model/output_Ch5/BiasAddBiasAdd!model/output_Ch5/MatMul:product:0/model/output_Ch5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch5/BiasAddћ
model/output_Ch5/SoftmaxSoftmax!model/output_Ch5/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch5/Softmax┬
&model/output_Ch4/MatMul/ReadVariableOpReadVariableOp/model_output_ch4_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch4/MatMul/ReadVariableOpЙ
model/output_Ch4/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch4/MatMul┐
'model/output_Ch4/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch4_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch4/BiasAdd/ReadVariableOp┼
model/output_Ch4/BiasAddBiasAdd!model/output_Ch4/MatMul:product:0/model/output_Ch4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch4/BiasAddћ
model/output_Ch4/SoftmaxSoftmax!model/output_Ch4/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch4/Softmax┬
&model/output_Ch3/MatMul/ReadVariableOpReadVariableOp/model_output_ch3_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch3/MatMul/ReadVariableOpЙ
model/output_Ch3/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch3/MatMul┐
'model/output_Ch3/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch3_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch3/BiasAdd/ReadVariableOp┼
model/output_Ch3/BiasAddBiasAdd!model/output_Ch3/MatMul:product:0/model/output_Ch3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch3/BiasAddћ
model/output_Ch3/SoftmaxSoftmax!model/output_Ch3/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch3/Softmax┬
&model/output_Ch2/MatMul/ReadVariableOpReadVariableOp/model_output_ch2_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch2/MatMul/ReadVariableOpЙ
model/output_Ch2/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch2/MatMul┐
'model/output_Ch2/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch2/BiasAdd/ReadVariableOp┼
model/output_Ch2/BiasAddBiasAdd!model/output_Ch2/MatMul:product:0/model/output_Ch2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch2/BiasAddћ
model/output_Ch2/SoftmaxSoftmax!model/output_Ch2/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch2/Softmax┬
&model/output_Ch1/MatMul/ReadVariableOpReadVariableOp/model_output_ch1_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02(
&model/output_Ch1/MatMul/ReadVariableOpЙ
model/output_Ch1/MatMulMatMulmodel/flatten/Reshape:output:0.model/output_Ch1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch1/MatMul┐
'model/output_Ch1/BiasAdd/ReadVariableOpReadVariableOp0model_output_ch1_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02)
'model/output_Ch1/BiasAdd/ReadVariableOp┼
model/output_Ch1/BiasAddBiasAdd!model/output_Ch1/MatMul:product:0/model/output_Ch1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Ch1/BiasAddћ
model/output_Ch1/SoftmaxSoftmax!model/output_Ch1/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Ch1/Softmax┐
%model/output_Zh/MatMul/ReadVariableOpReadVariableOp.model_output_zh_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02'
%model/output_Zh/MatMul/ReadVariableOp╗
model/output_Zh/MatMulMatMulmodel/flatten/Reshape:output:0-model/output_Zh/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Zh/MatMul╝
&model/output_Zh/BiasAdd/ReadVariableOpReadVariableOp/model_output_zh_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02(
&model/output_Zh/BiasAdd/ReadVariableOp┴
model/output_Zh/BiasAddBiasAdd model/output_Zh/MatMul:product:0.model/output_Zh/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
model/output_Zh/BiasAddЉ
model/output_Zh/SoftmaxSoftmax model/output_Zh/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
model/output_Zh/Softmaxб
IdentityIdentity"model/output_Ch1/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identityд

Identity_1Identity"model/output_Ch2/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_1д

Identity_2Identity"model/output_Ch3/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_2д

Identity_3Identity"model/output_Ch4/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_3д

Identity_4Identity"model/output_Ch5/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_4д

Identity_5Identity"model/output_Ch6/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_5Ц

Identity_6Identity!model/output_Zh/Softmax:softmax:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp(^model/output_Ch1/BiasAdd/ReadVariableOp'^model/output_Ch1/MatMul/ReadVariableOp(^model/output_Ch2/BiasAdd/ReadVariableOp'^model/output_Ch2/MatMul/ReadVariableOp(^model/output_Ch3/BiasAdd/ReadVariableOp'^model/output_Ch3/MatMul/ReadVariableOp(^model/output_Ch4/BiasAdd/ReadVariableOp'^model/output_Ch4/MatMul/ReadVariableOp(^model/output_Ch5/BiasAdd/ReadVariableOp'^model/output_Ch5/MatMul/ReadVariableOp(^model/output_Ch6/BiasAdd/ReadVariableOp'^model/output_Ch6/MatMul/ReadVariableOp'^model/output_Zh/BiasAdd/ReadVariableOp&^model/output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2R
'model/output_Ch1/BiasAdd/ReadVariableOp'model/output_Ch1/BiasAdd/ReadVariableOp2P
&model/output_Ch1/MatMul/ReadVariableOp&model/output_Ch1/MatMul/ReadVariableOp2R
'model/output_Ch2/BiasAdd/ReadVariableOp'model/output_Ch2/BiasAdd/ReadVariableOp2P
&model/output_Ch2/MatMul/ReadVariableOp&model/output_Ch2/MatMul/ReadVariableOp2R
'model/output_Ch3/BiasAdd/ReadVariableOp'model/output_Ch3/BiasAdd/ReadVariableOp2P
&model/output_Ch3/MatMul/ReadVariableOp&model/output_Ch3/MatMul/ReadVariableOp2R
'model/output_Ch4/BiasAdd/ReadVariableOp'model/output_Ch4/BiasAdd/ReadVariableOp2P
&model/output_Ch4/MatMul/ReadVariableOp&model/output_Ch4/MatMul/ReadVariableOp2R
'model/output_Ch5/BiasAdd/ReadVariableOp'model/output_Ch5/BiasAdd/ReadVariableOp2P
&model/output_Ch5/MatMul/ReadVariableOp&model/output_Ch5/MatMul/ReadVariableOp2R
'model/output_Ch6/BiasAdd/ReadVariableOp'model/output_Ch6/BiasAdd/ReadVariableOp2P
&model/output_Ch6/MatMul/ReadVariableOp&model/output_Ch6/MatMul/ReadVariableOp2P
&model/output_Zh/BiasAdd/ReadVariableOp&model/output_Zh/BiasAdd/ReadVariableOp2N
%model/output_Zh/MatMul/ReadVariableOp%model/output_Zh/MatMul/ReadVariableOp:% !

_user_specified_nameinput
Я
D
(__inference_flatten_layer_call_fn_161351

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1605572
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
х
a
C__inference_dropout_layer_call_and_return_conditional_losses_160492

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@:& "
 
_user_specified_nameinputs
В
D
(__inference_dropout_layer_call_fn_161305

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604922
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:& "
 
_user_specified_nameinputs
х
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_160420

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch2_layer_call_and_return_conditional_losses_161398

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ч
г
+__inference_output_Ch5_layer_call_fn_161459

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch5_layer_call_and_return_conditional_losses_1605992
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ж
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_161325

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
У
П
D__inference_conv2d_2_layer_call_and_return_conditional_losses_160439

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
У
П
D__inference_conv2d_1_layer_call_and_return_conditional_losses_160406

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
┌
Ў
&__inference_model_layer_call_fn_160946	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1609112
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2њ

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3њ

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4њ

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5њ

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameinput
М	
▀
F__inference_output_Ch4_layer_call_and_return_conditional_losses_160622

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_160557

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     @  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђђ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
У
b
C__inference_dropout_layer_call_and_return_conditional_losses_161290

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @@2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:& "
 
_user_specified_nameinputs
Э
a
(__inference_dropout_layer_call_fn_161300

inputs
identityѕбStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604872
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch1_layer_call_and_return_conditional_losses_160691

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ћe
┬
A__inference_model_layer_call_and_return_conditional_losses_160779	
input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2-
)output_ch6_statefulpartitionedcall_args_1-
)output_ch6_statefulpartitionedcall_args_2-
)output_ch5_statefulpartitionedcall_args_1-
)output_ch5_statefulpartitionedcall_args_2-
)output_ch4_statefulpartitionedcall_args_1-
)output_ch4_statefulpartitionedcall_args_2-
)output_ch3_statefulpartitionedcall_args_1-
)output_ch3_statefulpartitionedcall_args_2-
)output_ch2_statefulpartitionedcall_args_1-
)output_ch2_statefulpartitionedcall_args_2-
)output_ch1_statefulpartitionedcall_args_1-
)output_ch1_statefulpartitionedcall_args_2,
(output_zh_statefulpartitionedcall_args_1,
(output_zh_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб"output_Ch1/StatefulPartitionedCallб"output_Ch2/StatefulPartitionedCallб"output_Ch3/StatefulPartitionedCallб"output_Ch4/StatefulPartitionedCallб"output_Ch5/StatefulPartitionedCallб"output_Ch6/StatefulPartitionedCallб!output_Zh/StatefulPartitionedCallг
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1603732 
conv2d/StatefulPartitionedCallщ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1603872
max_pooling2d/PartitionedCallТ
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604922
dropout/PartitionedCall¤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1604062"
 conv2d_1/StatefulPartitionedCallЂ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1604202!
max_pooling2d_1/PartitionedCallО
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1604392"
 conv2d_2/StatefulPartitionedCallЂ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1604532!
max_pooling2d_2/PartitionedCallЬ
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605382
dropout_1/PartitionedCall▄
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1605572
flatten/PartitionedCallЛ
"output_Ch6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch6_statefulpartitionedcall_args_1)output_ch6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch6_layer_call_and_return_conditional_losses_1605762$
"output_Ch6/StatefulPartitionedCallЛ
"output_Ch5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch5_statefulpartitionedcall_args_1)output_ch5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch5_layer_call_and_return_conditional_losses_1605992$
"output_Ch5/StatefulPartitionedCallЛ
"output_Ch4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch4_statefulpartitionedcall_args_1)output_ch4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch4_layer_call_and_return_conditional_losses_1606222$
"output_Ch4/StatefulPartitionedCallЛ
"output_Ch3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch3_statefulpartitionedcall_args_1)output_ch3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch3_layer_call_and_return_conditional_losses_1606452$
"output_Ch3/StatefulPartitionedCallЛ
"output_Ch2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch2_statefulpartitionedcall_args_1)output_ch2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch2_layer_call_and_return_conditional_losses_1606682$
"output_Ch2/StatefulPartitionedCallЛ
"output_Ch1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch1_statefulpartitionedcall_args_1)output_ch1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch1_layer_call_and_return_conditional_losses_1606912$
"output_Ch1/StatefulPartitionedCall╠
!output_Zh/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(output_zh_statefulpartitionedcall_args_1(output_zh_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_output_Zh_layer_call_and_return_conditional_losses_1607142#
!output_Zh/StatefulPartitionedCallу
IdentityIdentity*output_Zh/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

IdentityВ

Identity_1Identity+output_Ch1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1В

Identity_2Identity+output_Ch2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2В

Identity_3Identity+output_Ch3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3В

Identity_4Identity+output_Ch4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4В

Identity_5Identity+output_Ch5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5В

Identity_6Identity+output_Ch6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2H
"output_Ch1/StatefulPartitionedCall"output_Ch1/StatefulPartitionedCall2H
"output_Ch2/StatefulPartitionedCall"output_Ch2/StatefulPartitionedCall2H
"output_Ch3/StatefulPartitionedCall"output_Ch3/StatefulPartitionedCall2H
"output_Ch4/StatefulPartitionedCall"output_Ch4/StatefulPartitionedCall2H
"output_Ch5/StatefulPartitionedCall"output_Ch5/StatefulPartitionedCall2H
"output_Ch6/StatefulPartitionedCall"output_Ch6/StatefulPartitionedCall2F
!output_Zh/StatefulPartitionedCall!output_Zh/StatefulPartitionedCall:% !

_user_specified_nameinput
М	
▀
F__inference_output_Ch1_layer_call_and_return_conditional_losses_161380

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
и╣
о(
"__inference__traced_restore_161993
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias'
#assignvariableop_6_output_zh_kernel%
!assignvariableop_7_output_zh_bias(
$assignvariableop_8_output_ch1_kernel&
"assignvariableop_9_output_ch1_bias)
%assignvariableop_10_output_ch2_kernel'
#assignvariableop_11_output_ch2_bias)
%assignvariableop_12_output_ch3_kernel'
#assignvariableop_13_output_ch3_bias)
%assignvariableop_14_output_ch4_kernel'
#assignvariableop_15_output_ch4_bias)
%assignvariableop_16_output_ch5_kernel'
#assignvariableop_17_output_ch5_bias)
%assignvariableop_18_output_ch6_kernel'
#assignvariableop_19_output_ch6_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1
assignvariableop_29_total_2
assignvariableop_30_count_2
assignvariableop_31_total_3
assignvariableop_32_count_3
assignvariableop_33_total_4
assignvariableop_34_count_4
assignvariableop_35_total_5
assignvariableop_36_count_5
assignvariableop_37_total_6
assignvariableop_38_count_6,
(assignvariableop_39_adam_conv2d_kernel_m*
&assignvariableop_40_adam_conv2d_bias_m.
*assignvariableop_41_adam_conv2d_1_kernel_m,
(assignvariableop_42_adam_conv2d_1_bias_m.
*assignvariableop_43_adam_conv2d_2_kernel_m,
(assignvariableop_44_adam_conv2d_2_bias_m/
+assignvariableop_45_adam_output_zh_kernel_m-
)assignvariableop_46_adam_output_zh_bias_m0
,assignvariableop_47_adam_output_ch1_kernel_m.
*assignvariableop_48_adam_output_ch1_bias_m0
,assignvariableop_49_adam_output_ch2_kernel_m.
*assignvariableop_50_adam_output_ch2_bias_m0
,assignvariableop_51_adam_output_ch3_kernel_m.
*assignvariableop_52_adam_output_ch3_bias_m0
,assignvariableop_53_adam_output_ch4_kernel_m.
*assignvariableop_54_adam_output_ch4_bias_m0
,assignvariableop_55_adam_output_ch5_kernel_m.
*assignvariableop_56_adam_output_ch5_bias_m0
,assignvariableop_57_adam_output_ch6_kernel_m.
*assignvariableop_58_adam_output_ch6_bias_m,
(assignvariableop_59_adam_conv2d_kernel_v*
&assignvariableop_60_adam_conv2d_bias_v.
*assignvariableop_61_adam_conv2d_1_kernel_v,
(assignvariableop_62_adam_conv2d_1_bias_v.
*assignvariableop_63_adam_conv2d_2_kernel_v,
(assignvariableop_64_adam_conv2d_2_bias_v/
+assignvariableop_65_adam_output_zh_kernel_v-
)assignvariableop_66_adam_output_zh_bias_v0
,assignvariableop_67_adam_output_ch1_kernel_v.
*assignvariableop_68_adam_output_ch1_bias_v0
,assignvariableop_69_adam_output_ch2_kernel_v.
*assignvariableop_70_adam_output_ch2_bias_v0
,assignvariableop_71_adam_output_ch3_kernel_v.
*assignvariableop_72_adam_output_ch3_bias_v0
,assignvariableop_73_adam_output_ch4_kernel_v.
*assignvariableop_74_adam_output_ch4_bias_v0
,assignvariableop_75_adam_output_ch5_kernel_v.
*assignvariableop_76_adam_output_ch5_bias_v0
,assignvariableop_77_adam_output_ch6_kernel_v.
*assignvariableop_78_adam_output_ch6_bias_v
identity_80ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1д+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*▓*
valueе*BЦ*OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*│
valueЕBдOB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╣
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*м
_output_shapes┐
╝:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identityј
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1ћ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ў
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3ќ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4ў
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5ќ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ў
AssignVariableOp_6AssignVariableOp#assignvariableop_6_output_zh_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ќ
AssignVariableOp_7AssignVariableOp!assignvariableop_7_output_zh_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8џ
AssignVariableOp_8AssignVariableOp$assignvariableop_8_output_ch1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9ў
AssignVariableOp_9AssignVariableOp"assignvariableop_9_output_ch1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10ъ
AssignVariableOp_10AssignVariableOp%assignvariableop_10_output_ch2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11ю
AssignVariableOp_11AssignVariableOp#assignvariableop_11_output_ch2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12ъ
AssignVariableOp_12AssignVariableOp%assignvariableop_12_output_ch3_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13ю
AssignVariableOp_13AssignVariableOp#assignvariableop_13_output_ch3_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14ъ
AssignVariableOp_14AssignVariableOp%assignvariableop_14_output_ch4_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15ю
AssignVariableOp_15AssignVariableOp#assignvariableop_15_output_ch4_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16ъ
AssignVariableOp_16AssignVariableOp%assignvariableop_16_output_ch5_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17ю
AssignVariableOp_17AssignVariableOp#assignvariableop_17_output_ch5_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18ъ
AssignVariableOp_18AssignVariableOp%assignvariableop_18_output_ch6_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19ю
AssignVariableOp_19AssignVariableOp#assignvariableop_19_output_ch6_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0	*
_output_shapes
:2
Identity_20ќ
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21ў
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22ў
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ќ
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ъ
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25њ
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26њ
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27ћ
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28ћ
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29ћ
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_2Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30ћ
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_2Identity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31ћ
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_3Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32ћ
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_3Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33ћ
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_4Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34ћ
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_4Identity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35ћ
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_5Identity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36ћ
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_5Identity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37ћ
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_6Identity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38ћ
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_6Identity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39А
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv2d_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Ъ
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv2d_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Б
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42А
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Б
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_2_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44А
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_2_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45ц
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_output_zh_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_output_zh_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Ц
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_output_ch1_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48Б
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_output_ch1_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49Ц
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_output_ch2_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50Б
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_output_ch2_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51Ц
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_output_ch3_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52Б
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_output_ch3_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53Ц
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_output_ch4_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54Б
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_output_ch4_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Ц
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_output_ch5_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Б
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_output_ch5_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57Ц
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_output_ch6_kernel_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58Б
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_output_ch6_bias_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59А
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_conv2d_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Ъ
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_conv2d_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61Б
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_1_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62А
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_1_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63Б
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_2_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64А
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_2_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65ц
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_output_zh_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66б
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_output_zh_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67Ц
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_output_ch1_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68Б
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_output_ch1_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69Ц
AssignVariableOp_69AssignVariableOp,assignvariableop_69_adam_output_ch2_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70Б
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_output_ch2_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71Ц
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_output_ch3_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72Б
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_output_ch3_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73Ц
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_output_ch4_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74Б
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_output_ch4_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75Ц
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_output_ch5_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76Б
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_output_ch5_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77Ц
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_output_ch6_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78Б
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_output_ch6_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpе
Identity_79Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_79х
Identity_80IdentityIdentity_79:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_80"#
identity_80Identity_80:output:0*М
_input_shapes┴
Й: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
М	
▀
F__inference_output_Ch6_layer_call_and_return_conditional_losses_161470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ЏЃ
В
__inference__traced_save_161744
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop/
+savev2_output_zh_kernel_read_readvariableop-
)savev2_output_zh_bias_read_readvariableop0
,savev2_output_ch1_kernel_read_readvariableop.
*savev2_output_ch1_bias_read_readvariableop0
,savev2_output_ch2_kernel_read_readvariableop.
*savev2_output_ch2_bias_read_readvariableop0
,savev2_output_ch3_kernel_read_readvariableop.
*savev2_output_ch3_bias_read_readvariableop0
,savev2_output_ch4_kernel_read_readvariableop.
*savev2_output_ch4_bias_read_readvariableop0
,savev2_output_ch5_kernel_read_readvariableop.
*savev2_output_ch5_bias_read_readvariableop0
,savev2_output_ch6_kernel_read_readvariableop.
*savev2_output_ch6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop6
2savev2_adam_output_zh_kernel_m_read_readvariableop4
0savev2_adam_output_zh_bias_m_read_readvariableop7
3savev2_adam_output_ch1_kernel_m_read_readvariableop5
1savev2_adam_output_ch1_bias_m_read_readvariableop7
3savev2_adam_output_ch2_kernel_m_read_readvariableop5
1savev2_adam_output_ch2_bias_m_read_readvariableop7
3savev2_adam_output_ch3_kernel_m_read_readvariableop5
1savev2_adam_output_ch3_bias_m_read_readvariableop7
3savev2_adam_output_ch4_kernel_m_read_readvariableop5
1savev2_adam_output_ch4_bias_m_read_readvariableop7
3savev2_adam_output_ch5_kernel_m_read_readvariableop5
1savev2_adam_output_ch5_bias_m_read_readvariableop7
3savev2_adam_output_ch6_kernel_m_read_readvariableop5
1savev2_adam_output_ch6_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop6
2savev2_adam_output_zh_kernel_v_read_readvariableop4
0savev2_adam_output_zh_bias_v_read_readvariableop7
3savev2_adam_output_ch1_kernel_v_read_readvariableop5
1savev2_adam_output_ch1_bias_v_read_readvariableop7
3savev2_adam_output_ch2_kernel_v_read_readvariableop5
1savev2_adam_output_ch2_bias_v_read_readvariableop7
3savev2_adam_output_ch3_kernel_v_read_readvariableop5
1savev2_adam_output_ch3_bias_v_read_readvariableop7
3savev2_adam_output_ch4_kernel_v_read_readvariableop5
1savev2_adam_output_ch4_bias_v_read_readvariableop7
3savev2_adam_output_ch5_kernel_v_read_readvariableop5
1savev2_adam_output_ch5_bias_v_read_readvariableop7
3savev2_adam_output_ch6_kernel_v_read_readvariableop5
1savev2_adam_output_ch6_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ц
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_086a39e806e74dd69008a5de5bf4f706/part2
StringJoin/inputs_1Ђ

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

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
ShardedFilenameа+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*▓*
valueе*BЦ*OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*│
valueЕBдOB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesе
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop+savev2_output_zh_kernel_read_readvariableop)savev2_output_zh_bias_read_readvariableop,savev2_output_ch1_kernel_read_readvariableop*savev2_output_ch1_bias_read_readvariableop,savev2_output_ch2_kernel_read_readvariableop*savev2_output_ch2_bias_read_readvariableop,savev2_output_ch3_kernel_read_readvariableop*savev2_output_ch3_bias_read_readvariableop,savev2_output_ch4_kernel_read_readvariableop*savev2_output_ch4_bias_read_readvariableop,savev2_output_ch5_kernel_read_readvariableop*savev2_output_ch5_bias_read_readvariableop,savev2_output_ch6_kernel_read_readvariableop*savev2_output_ch6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop2savev2_adam_output_zh_kernel_m_read_readvariableop0savev2_adam_output_zh_bias_m_read_readvariableop3savev2_adam_output_ch1_kernel_m_read_readvariableop1savev2_adam_output_ch1_bias_m_read_readvariableop3savev2_adam_output_ch2_kernel_m_read_readvariableop1savev2_adam_output_ch2_bias_m_read_readvariableop3savev2_adam_output_ch3_kernel_m_read_readvariableop1savev2_adam_output_ch3_bias_m_read_readvariableop3savev2_adam_output_ch4_kernel_m_read_readvariableop1savev2_adam_output_ch4_bias_m_read_readvariableop3savev2_adam_output_ch5_kernel_m_read_readvariableop1savev2_adam_output_ch5_bias_m_read_readvariableop3savev2_adam_output_ch6_kernel_m_read_readvariableop1savev2_adam_output_ch6_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop2savev2_adam_output_zh_kernel_v_read_readvariableop0savev2_adam_output_zh_bias_v_read_readvariableop3savev2_adam_output_ch1_kernel_v_read_readvariableop1savev2_adam_output_ch1_bias_v_read_readvariableop3savev2_adam_output_ch2_kernel_v_read_readvariableop1savev2_adam_output_ch2_bias_v_read_readvariableop3savev2_adam_output_ch3_kernel_v_read_readvariableop1savev2_adam_output_ch3_bias_v_read_readvariableop3savev2_adam_output_ch4_kernel_v_read_readvariableop1savev2_adam_output_ch4_bias_v_read_readvariableop3savev2_adam_output_ch5_kernel_v_read_readvariableop1savev2_adam_output_ch5_bias_v_read_readvariableop3savev2_adam_output_ch6_kernel_v_read_readvariableop1savev2_adam_output_ch6_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *]
dtypesS
Q2O	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Љ
_input_shapes 
Ч: ::: : : @:@:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A: : : : : : : : : : : : : : : : : : : ::: : : @:@:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A::: : : @:@:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A:
ђђA:A: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_161346

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     @  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђђ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
Ж
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_160533

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
│k
ѕ
A__inference_model_layer_call_and_return_conditional_losses_160733	
input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2-
)output_ch6_statefulpartitionedcall_args_1-
)output_ch6_statefulpartitionedcall_args_2-
)output_ch5_statefulpartitionedcall_args_1-
)output_ch5_statefulpartitionedcall_args_2-
)output_ch4_statefulpartitionedcall_args_1-
)output_ch4_statefulpartitionedcall_args_2-
)output_ch3_statefulpartitionedcall_args_1-
)output_ch3_statefulpartitionedcall_args_2-
)output_ch2_statefulpartitionedcall_args_1-
)output_ch2_statefulpartitionedcall_args_2-
)output_ch1_statefulpartitionedcall_args_1-
)output_ch1_statefulpartitionedcall_args_2,
(output_zh_statefulpartitionedcall_args_1,
(output_zh_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб"output_Ch1/StatefulPartitionedCallб"output_Ch2/StatefulPartitionedCallб"output_Ch3/StatefulPartitionedCallб"output_Ch4/StatefulPartitionedCallб"output_Ch5/StatefulPartitionedCallб"output_Ch6/StatefulPartitionedCallб!output_Zh/StatefulPartitionedCallг
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1603732 
conv2d/StatefulPartitionedCallщ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1603872
max_pooling2d/PartitionedCall■
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604872!
dropout/StatefulPartitionedCallО
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1604062"
 conv2d_1/StatefulPartitionedCallЂ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1604202!
max_pooling2d_1/PartitionedCallО
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1604392"
 conv2d_2/StatefulPartitionedCallЂ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1604532!
max_pooling2d_2/PartitionedCallе
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605332#
!dropout_1/StatefulPartitionedCallС
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1605572
flatten/PartitionedCallЛ
"output_Ch6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch6_statefulpartitionedcall_args_1)output_ch6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch6_layer_call_and_return_conditional_losses_1605762$
"output_Ch6/StatefulPartitionedCallЛ
"output_Ch5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch5_statefulpartitionedcall_args_1)output_ch5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch5_layer_call_and_return_conditional_losses_1605992$
"output_Ch5/StatefulPartitionedCallЛ
"output_Ch4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch4_statefulpartitionedcall_args_1)output_ch4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch4_layer_call_and_return_conditional_losses_1606222$
"output_Ch4/StatefulPartitionedCallЛ
"output_Ch3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch3_statefulpartitionedcall_args_1)output_ch3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch3_layer_call_and_return_conditional_losses_1606452$
"output_Ch3/StatefulPartitionedCallЛ
"output_Ch2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch2_statefulpartitionedcall_args_1)output_ch2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch2_layer_call_and_return_conditional_losses_1606682$
"output_Ch2/StatefulPartitionedCallЛ
"output_Ch1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch1_statefulpartitionedcall_args_1)output_ch1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch1_layer_call_and_return_conditional_losses_1606912$
"output_Ch1/StatefulPartitionedCall╠
!output_Zh/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(output_zh_statefulpartitionedcall_args_1(output_zh_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_output_Zh_layer_call_and_return_conditional_losses_1607142#
!output_Zh/StatefulPartitionedCallГ
IdentityIdentity*output_Zh/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity▓

Identity_1Identity+output_Ch1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1▓

Identity_2Identity+output_Ch2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2▓

Identity_3Identity+output_Ch3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3▓

Identity_4Identity+output_Ch4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4▓

Identity_5Identity+output_Ch5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5▓

Identity_6Identity+output_Ch6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"output_Ch1/StatefulPartitionedCall"output_Ch1/StatefulPartitionedCall2H
"output_Ch2/StatefulPartitionedCall"output_Ch2/StatefulPartitionedCall2H
"output_Ch3/StatefulPartitionedCall"output_Ch3/StatefulPartitionedCall2H
"output_Ch4/StatefulPartitionedCall"output_Ch4/StatefulPartitionedCall2H
"output_Ch5/StatefulPartitionedCall"output_Ch5/StatefulPartitionedCall2H
"output_Ch6/StatefulPartitionedCall"output_Ch6/StatefulPartitionedCall2F
!output_Zh/StatefulPartitionedCall!output_Zh/StatefulPartitionedCall:% !

_user_specified_nameinput
ч
г
+__inference_output_Ch1_layer_call_fn_161387

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch1_layer_call_and_return_conditional_losses_1606912
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
├
ф
)__inference_conv2d_1_layer_call_fn_160414

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1604062
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
­њ
ў
A__inference_model_layer_call_and_return_conditional_losses_161196

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)output_ch6_matmul_readvariableop_resource.
*output_ch6_biasadd_readvariableop_resource-
)output_ch5_matmul_readvariableop_resource.
*output_ch5_biasadd_readvariableop_resource-
)output_ch4_matmul_readvariableop_resource.
*output_ch4_biasadd_readvariableop_resource-
)output_ch3_matmul_readvariableop_resource.
*output_ch3_biasadd_readvariableop_resource-
)output_ch2_matmul_readvariableop_resource.
*output_ch2_biasadd_readvariableop_resource-
)output_ch1_matmul_readvariableop_resource.
*output_ch1_biasadd_readvariableop_resource,
(output_zh_matmul_readvariableop_resource-
)output_zh_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpб!output_Ch1/BiasAdd/ReadVariableOpб output_Ch1/MatMul/ReadVariableOpб!output_Ch2/BiasAdd/ReadVariableOpб output_Ch2/MatMul/ReadVariableOpб!output_Ch3/BiasAdd/ReadVariableOpб output_Ch3/MatMul/ReadVariableOpб!output_Ch4/BiasAdd/ReadVariableOpб output_Ch4/MatMul/ReadVariableOpб!output_Ch5/BiasAdd/ReadVariableOpб output_Ch5/MatMul/ReadVariableOpб!output_Ch6/BiasAdd/ReadVariableOpб output_Ch6/MatMul/ReadVariableOpб output_Zh/BiasAdd/ReadVariableOpбoutput_Zh/MatMul/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp║
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
conv2d/Relu┴
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolі
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @@2
dropout/Identity░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
conv2d_1/ReluК
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpп
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
conv2d_2/ReluК
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolљ
dropout_1/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @2
dropout_1/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     @  2
flatten/Constќ
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         ђђ2
flatten/Reshape░
 output_Ch6/MatMul/ReadVariableOpReadVariableOp)output_ch6_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch6/MatMul/ReadVariableOpд
output_Ch6/MatMulMatMulflatten/Reshape:output:0(output_Ch6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch6/MatMulГ
!output_Ch6/BiasAdd/ReadVariableOpReadVariableOp*output_ch6_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch6/BiasAdd/ReadVariableOpГ
output_Ch6/BiasAddBiasAddoutput_Ch6/MatMul:product:0)output_Ch6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch6/BiasAddѓ
output_Ch6/SoftmaxSoftmaxoutput_Ch6/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch6/Softmax░
 output_Ch5/MatMul/ReadVariableOpReadVariableOp)output_ch5_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch5/MatMul/ReadVariableOpд
output_Ch5/MatMulMatMulflatten/Reshape:output:0(output_Ch5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch5/MatMulГ
!output_Ch5/BiasAdd/ReadVariableOpReadVariableOp*output_ch5_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch5/BiasAdd/ReadVariableOpГ
output_Ch5/BiasAddBiasAddoutput_Ch5/MatMul:product:0)output_Ch5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch5/BiasAddѓ
output_Ch5/SoftmaxSoftmaxoutput_Ch5/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch5/Softmax░
 output_Ch4/MatMul/ReadVariableOpReadVariableOp)output_ch4_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch4/MatMul/ReadVariableOpд
output_Ch4/MatMulMatMulflatten/Reshape:output:0(output_Ch4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch4/MatMulГ
!output_Ch4/BiasAdd/ReadVariableOpReadVariableOp*output_ch4_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch4/BiasAdd/ReadVariableOpГ
output_Ch4/BiasAddBiasAddoutput_Ch4/MatMul:product:0)output_Ch4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch4/BiasAddѓ
output_Ch4/SoftmaxSoftmaxoutput_Ch4/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch4/Softmax░
 output_Ch3/MatMul/ReadVariableOpReadVariableOp)output_ch3_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch3/MatMul/ReadVariableOpд
output_Ch3/MatMulMatMulflatten/Reshape:output:0(output_Ch3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch3/MatMulГ
!output_Ch3/BiasAdd/ReadVariableOpReadVariableOp*output_ch3_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch3/BiasAdd/ReadVariableOpГ
output_Ch3/BiasAddBiasAddoutput_Ch3/MatMul:product:0)output_Ch3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch3/BiasAddѓ
output_Ch3/SoftmaxSoftmaxoutput_Ch3/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch3/Softmax░
 output_Ch2/MatMul/ReadVariableOpReadVariableOp)output_ch2_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch2/MatMul/ReadVariableOpд
output_Ch2/MatMulMatMulflatten/Reshape:output:0(output_Ch2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch2/MatMulГ
!output_Ch2/BiasAdd/ReadVariableOpReadVariableOp*output_ch2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch2/BiasAdd/ReadVariableOpГ
output_Ch2/BiasAddBiasAddoutput_Ch2/MatMul:product:0)output_Ch2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch2/BiasAddѓ
output_Ch2/SoftmaxSoftmaxoutput_Ch2/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch2/Softmax░
 output_Ch1/MatMul/ReadVariableOpReadVariableOp)output_ch1_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch1/MatMul/ReadVariableOpд
output_Ch1/MatMulMatMulflatten/Reshape:output:0(output_Ch1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch1/MatMulГ
!output_Ch1/BiasAdd/ReadVariableOpReadVariableOp*output_ch1_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch1/BiasAdd/ReadVariableOpГ
output_Ch1/BiasAddBiasAddoutput_Ch1/MatMul:product:0)output_Ch1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch1/BiasAddѓ
output_Ch1/SoftmaxSoftmaxoutput_Ch1/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch1/SoftmaxГ
output_Zh/MatMul/ReadVariableOpReadVariableOp(output_zh_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02!
output_Zh/MatMul/ReadVariableOpБ
output_Zh/MatMulMatMulflatten/Reshape:output:0'output_Zh/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Zh/MatMulф
 output_Zh/BiasAdd/ReadVariableOpReadVariableOp)output_zh_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02"
 output_Zh/BiasAdd/ReadVariableOpЕ
output_Zh/BiasAddBiasAddoutput_Zh/MatMul:product:0(output_Zh/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Zh/BiasAdd
output_Zh/SoftmaxSoftmaxoutput_Zh/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Zh/SoftmaxБ
IdentityIdentityoutput_Zh/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identityе

Identity_1Identityoutput_Ch1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_1е

Identity_2Identityoutput_Ch2/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_2е

Identity_3Identityoutput_Ch3/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_3е

Identity_4Identityoutput_Ch4/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_4е

Identity_5Identityoutput_Ch5/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_5е

Identity_6Identityoutput_Ch6/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2F
!output_Ch1/BiasAdd/ReadVariableOp!output_Ch1/BiasAdd/ReadVariableOp2D
 output_Ch1/MatMul/ReadVariableOp output_Ch1/MatMul/ReadVariableOp2F
!output_Ch2/BiasAdd/ReadVariableOp!output_Ch2/BiasAdd/ReadVariableOp2D
 output_Ch2/MatMul/ReadVariableOp output_Ch2/MatMul/ReadVariableOp2F
!output_Ch3/BiasAdd/ReadVariableOp!output_Ch3/BiasAdd/ReadVariableOp2D
 output_Ch3/MatMul/ReadVariableOp output_Ch3/MatMul/ReadVariableOp2F
!output_Ch4/BiasAdd/ReadVariableOp!output_Ch4/BiasAdd/ReadVariableOp2D
 output_Ch4/MatMul/ReadVariableOp output_Ch4/MatMul/ReadVariableOp2F
!output_Ch5/BiasAdd/ReadVariableOp!output_Ch5/BiasAdd/ReadVariableOp2D
 output_Ch5/MatMul/ReadVariableOp output_Ch5/MatMul/ReadVariableOp2F
!output_Ch6/BiasAdd/ReadVariableOp!output_Ch6/BiasAdd/ReadVariableOp2D
 output_Ch6/MatMul/ReadVariableOp output_Ch6/MatMul/ReadVariableOp2D
 output_Zh/BiasAdd/ReadVariableOp output_Zh/BiasAdd/ReadVariableOp2B
output_Zh/MatMul/ReadVariableOpoutput_Zh/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
и
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_161330

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_160387

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch6_layer_call_and_return_conditional_losses_160576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch2_layer_call_and_return_conditional_losses_160668

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ь║
ў
A__inference_model_layer_call_and_return_conditional_losses_161109

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)output_ch6_matmul_readvariableop_resource.
*output_ch6_biasadd_readvariableop_resource-
)output_ch5_matmul_readvariableop_resource.
*output_ch5_biasadd_readvariableop_resource-
)output_ch4_matmul_readvariableop_resource.
*output_ch4_biasadd_readvariableop_resource-
)output_ch3_matmul_readvariableop_resource.
*output_ch3_biasadd_readvariableop_resource-
)output_ch2_matmul_readvariableop_resource.
*output_ch2_biasadd_readvariableop_resource-
)output_ch1_matmul_readvariableop_resource.
*output_ch1_biasadd_readvariableop_resource,
(output_zh_matmul_readvariableop_resource-
)output_zh_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpб!output_Ch1/BiasAdd/ReadVariableOpб output_Ch1/MatMul/ReadVariableOpб!output_Ch2/BiasAdd/ReadVariableOpб output_Ch2/MatMul/ReadVariableOpб!output_Ch3/BiasAdd/ReadVariableOpб output_Ch3/MatMul/ReadVariableOpб!output_Ch4/BiasAdd/ReadVariableOpб output_Ch4/MatMul/ReadVariableOpб!output_Ch5/BiasAdd/ReadVariableOpб output_Ch5/MatMul/ReadVariableOpб!output_Ch6/BiasAdd/ReadVariableOpб output_Ch6/MatMul/ReadVariableOpб output_Zh/BiasAdd/ReadVariableOpбoutput_Zh/MatMul/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp║
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
conv2d/Relu┴
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout/dropout/rate|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЇ
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout/dropout/random_uniform/minЇ
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2$
"dropout/dropout/random_uniform/maxн
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform╩
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"dropout/dropout/random_uniform/subУ
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@2$
"dropout/dropout/random_uniform/mulо
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@2 
dropout/dropout/random_uniforms
dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/dropout/sub/xЉ
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/dropout/sub{
dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/dropout/truediv/xЏ
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/dropout/truediv╔
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:         @@2
dropout/dropout/GreaterEqualе
dropout/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:         @@2
dropout/dropout/mulЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
dropout/dropout/Castб
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2
dropout/dropout/mul_1░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
conv2d_1/ReluК
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpп
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:           @2
conv2d_2/ReluК
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolu
dropout_1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout_1/dropout/rateѓ
dropout_1/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeЉ
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_1/dropout/random_uniform/minЉ
$dropout_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2&
$dropout_1/dropout/random_uniform/max┌
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformм
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_1/dropout/random_uniform/sub­
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @2&
$dropout_1/dropout/random_uniform/mulя
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @2"
 dropout_1/dropout/random_uniformw
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout_1/dropout/sub/xЎ
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_1/dropout/sub
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout_1/dropout/truediv/xБ
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_1/dropout/truedivЛ
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*/
_output_shapes
:         @2 
dropout_1/dropout/GreaterEqual░
dropout_1/dropout/mulMul max_pooling2d_2/MaxPool:output:0dropout_1/dropout/truediv:z:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/mulЦ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_1/dropout/Castф
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     @  2
flatten/Constќ
flatten/ReshapeReshapedropout_1/dropout/mul_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         ђђ2
flatten/Reshape░
 output_Ch6/MatMul/ReadVariableOpReadVariableOp)output_ch6_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch6/MatMul/ReadVariableOpд
output_Ch6/MatMulMatMulflatten/Reshape:output:0(output_Ch6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch6/MatMulГ
!output_Ch6/BiasAdd/ReadVariableOpReadVariableOp*output_ch6_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch6/BiasAdd/ReadVariableOpГ
output_Ch6/BiasAddBiasAddoutput_Ch6/MatMul:product:0)output_Ch6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch6/BiasAddѓ
output_Ch6/SoftmaxSoftmaxoutput_Ch6/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch6/Softmax░
 output_Ch5/MatMul/ReadVariableOpReadVariableOp)output_ch5_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch5/MatMul/ReadVariableOpд
output_Ch5/MatMulMatMulflatten/Reshape:output:0(output_Ch5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch5/MatMulГ
!output_Ch5/BiasAdd/ReadVariableOpReadVariableOp*output_ch5_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch5/BiasAdd/ReadVariableOpГ
output_Ch5/BiasAddBiasAddoutput_Ch5/MatMul:product:0)output_Ch5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch5/BiasAddѓ
output_Ch5/SoftmaxSoftmaxoutput_Ch5/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch5/Softmax░
 output_Ch4/MatMul/ReadVariableOpReadVariableOp)output_ch4_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch4/MatMul/ReadVariableOpд
output_Ch4/MatMulMatMulflatten/Reshape:output:0(output_Ch4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch4/MatMulГ
!output_Ch4/BiasAdd/ReadVariableOpReadVariableOp*output_ch4_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch4/BiasAdd/ReadVariableOpГ
output_Ch4/BiasAddBiasAddoutput_Ch4/MatMul:product:0)output_Ch4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch4/BiasAddѓ
output_Ch4/SoftmaxSoftmaxoutput_Ch4/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch4/Softmax░
 output_Ch3/MatMul/ReadVariableOpReadVariableOp)output_ch3_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch3/MatMul/ReadVariableOpд
output_Ch3/MatMulMatMulflatten/Reshape:output:0(output_Ch3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch3/MatMulГ
!output_Ch3/BiasAdd/ReadVariableOpReadVariableOp*output_ch3_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch3/BiasAdd/ReadVariableOpГ
output_Ch3/BiasAddBiasAddoutput_Ch3/MatMul:product:0)output_Ch3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch3/BiasAddѓ
output_Ch3/SoftmaxSoftmaxoutput_Ch3/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch3/Softmax░
 output_Ch2/MatMul/ReadVariableOpReadVariableOp)output_ch2_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch2/MatMul/ReadVariableOpд
output_Ch2/MatMulMatMulflatten/Reshape:output:0(output_Ch2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch2/MatMulГ
!output_Ch2/BiasAdd/ReadVariableOpReadVariableOp*output_ch2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch2/BiasAdd/ReadVariableOpГ
output_Ch2/BiasAddBiasAddoutput_Ch2/MatMul:product:0)output_Ch2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch2/BiasAddѓ
output_Ch2/SoftmaxSoftmaxoutput_Ch2/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch2/Softmax░
 output_Ch1/MatMul/ReadVariableOpReadVariableOp)output_ch1_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02"
 output_Ch1/MatMul/ReadVariableOpд
output_Ch1/MatMulMatMulflatten/Reshape:output:0(output_Ch1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch1/MatMulГ
!output_Ch1/BiasAdd/ReadVariableOpReadVariableOp*output_ch1_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02#
!output_Ch1/BiasAdd/ReadVariableOpГ
output_Ch1/BiasAddBiasAddoutput_Ch1/MatMul:product:0)output_Ch1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Ch1/BiasAddѓ
output_Ch1/SoftmaxSoftmaxoutput_Ch1/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Ch1/SoftmaxГ
output_Zh/MatMul/ReadVariableOpReadVariableOp(output_zh_matmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02!
output_Zh/MatMul/ReadVariableOpБ
output_Zh/MatMulMatMulflatten/Reshape:output:0'output_Zh/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Zh/MatMulф
 output_Zh/BiasAdd/ReadVariableOpReadVariableOp)output_zh_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02"
 output_Zh/BiasAdd/ReadVariableOpЕ
output_Zh/BiasAddBiasAddoutput_Zh/MatMul:product:0(output_Zh/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
output_Zh/BiasAdd
output_Zh/SoftmaxSoftmaxoutput_Zh/BiasAdd:output:0*
T0*'
_output_shapes
:         A2
output_Zh/SoftmaxБ
IdentityIdentityoutput_Zh/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identityе

Identity_1Identityoutput_Ch1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_1е

Identity_2Identityoutput_Ch2/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_2е

Identity_3Identityoutput_Ch3/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_3е

Identity_4Identityoutput_Ch4/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_4е

Identity_5Identityoutput_Ch5/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_5е

Identity_6Identityoutput_Ch6/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp"^output_Ch1/BiasAdd/ReadVariableOp!^output_Ch1/MatMul/ReadVariableOp"^output_Ch2/BiasAdd/ReadVariableOp!^output_Ch2/MatMul/ReadVariableOp"^output_Ch3/BiasAdd/ReadVariableOp!^output_Ch3/MatMul/ReadVariableOp"^output_Ch4/BiasAdd/ReadVariableOp!^output_Ch4/MatMul/ReadVariableOp"^output_Ch5/BiasAdd/ReadVariableOp!^output_Ch5/MatMul/ReadVariableOp"^output_Ch6/BiasAdd/ReadVariableOp!^output_Ch6/MatMul/ReadVariableOp!^output_Zh/BiasAdd/ReadVariableOp ^output_Zh/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2F
!output_Ch1/BiasAdd/ReadVariableOp!output_Ch1/BiasAdd/ReadVariableOp2D
 output_Ch1/MatMul/ReadVariableOp output_Ch1/MatMul/ReadVariableOp2F
!output_Ch2/BiasAdd/ReadVariableOp!output_Ch2/BiasAdd/ReadVariableOp2D
 output_Ch2/MatMul/ReadVariableOp output_Ch2/MatMul/ReadVariableOp2F
!output_Ch3/BiasAdd/ReadVariableOp!output_Ch3/BiasAdd/ReadVariableOp2D
 output_Ch3/MatMul/ReadVariableOp output_Ch3/MatMul/ReadVariableOp2F
!output_Ch4/BiasAdd/ReadVariableOp!output_Ch4/BiasAdd/ReadVariableOp2D
 output_Ch4/MatMul/ReadVariableOp output_Ch4/MatMul/ReadVariableOp2F
!output_Ch5/BiasAdd/ReadVariableOp!output_Ch5/BiasAdd/ReadVariableOp2D
 output_Ch5/MatMul/ReadVariableOp output_Ch5/MatMul/ReadVariableOp2F
!output_Ch6/BiasAdd/ReadVariableOp!output_Ch6/BiasAdd/ReadVariableOp2D
 output_Ch6/MatMul/ReadVariableOp output_Ch6/MatMul/ReadVariableOp2D
 output_Zh/BiasAdd/ReadVariableOp output_Zh/BiasAdd/ReadVariableOp2B
output_Zh/MatMul/ReadVariableOpoutput_Zh/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
х
a
C__inference_dropout_layer_call_and_return_conditional_losses_161295

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@:& "
 
_user_specified_nameinputs
П
џ
&__inference_model_layer_call_fn_161270

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1609112
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2њ

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3њ

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4њ

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5њ

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┐
е
'__inference_conv2d_layer_call_fn_160381

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1603732
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Т
█
B__inference_conv2d_layer_call_and_return_conditional_losses_160373

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpх
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpџ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
­
F
*__inference_dropout_1_layer_call_fn_161340

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
щ
Ф
*__inference_output_Zh_layer_call_fn_161369

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_output_Zh_layer_call_and_return_conditional_losses_1607142
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch3_layer_call_and_return_conditional_losses_161416

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ч
г
+__inference_output_Ch4_layer_call_fn_161441

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch4_layer_call_and_return_conditional_losses_1606222
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
И
Ќ
$__inference_signature_wrapper_160992	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_1603602
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2њ

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3њ

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4њ

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5њ

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameinput
и
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_160538

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
┌
Ў
&__inference_model_layer_call_fn_160863	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1608282
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2њ

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3њ

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4њ

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5њ

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameinput
м	
я
E__inference_output_Zh_layer_call_and_return_conditional_losses_160714

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch4_layer_call_and_return_conditional_losses_161434

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch5_layer_call_and_return_conditional_losses_161452

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
У
b
C__inference_dropout_layer_call_and_return_conditional_losses_160487

inputs
identityѕa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02&
$dropout/random_uniform/RandomUniformф
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@2
dropout/random_uniform/mulХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivЕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @@2
dropout/mulЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
dropout/Castѓ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch3_layer_call_and_return_conditional_losses_160645

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
х
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_160453

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╬
L
0__inference_max_pooling2d_1_layer_call_fn_160426

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1604202
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ч
г
+__inference_output_Ch6_layer_call_fn_161477

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch6_layer_call_and_return_conditional_losses_1605762
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
П
џ
&__inference_model_layer_call_fn_161233

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
	2*,
_gradient_op_typePartitionedCallUnused*Џ
_output_shapesѕ
Ё:         A:         A:         A:         A:         A:         A:         A*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1608282
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2њ

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3њ

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4њ

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5њ

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ўe
├
A__inference_model_layer_call_and_return_conditional_losses_160911

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2-
)output_ch6_statefulpartitionedcall_args_1-
)output_ch6_statefulpartitionedcall_args_2-
)output_ch5_statefulpartitionedcall_args_1-
)output_ch5_statefulpartitionedcall_args_2-
)output_ch4_statefulpartitionedcall_args_1-
)output_ch4_statefulpartitionedcall_args_2-
)output_ch3_statefulpartitionedcall_args_1-
)output_ch3_statefulpartitionedcall_args_2-
)output_ch2_statefulpartitionedcall_args_1-
)output_ch2_statefulpartitionedcall_args_2-
)output_ch1_statefulpartitionedcall_args_1-
)output_ch1_statefulpartitionedcall_args_2,
(output_zh_statefulpartitionedcall_args_1,
(output_zh_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб"output_Ch1/StatefulPartitionedCallб"output_Ch2/StatefulPartitionedCallб"output_Ch3/StatefulPartitionedCallб"output_Ch4/StatefulPartitionedCallб"output_Ch5/StatefulPartitionedCallб"output_Ch6/StatefulPartitionedCallб!output_Zh/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1603732 
conv2d/StatefulPartitionedCallщ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1603872
max_pooling2d/PartitionedCallТ
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604922
dropout/PartitionedCall¤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1604062"
 conv2d_1/StatefulPartitionedCallЂ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1604202!
max_pooling2d_1/PartitionedCallО
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1604392"
 conv2d_2/StatefulPartitionedCallЂ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1604532!
max_pooling2d_2/PartitionedCallЬ
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605382
dropout_1/PartitionedCall▄
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1605572
flatten/PartitionedCallЛ
"output_Ch6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch6_statefulpartitionedcall_args_1)output_ch6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch6_layer_call_and_return_conditional_losses_1605762$
"output_Ch6/StatefulPartitionedCallЛ
"output_Ch5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch5_statefulpartitionedcall_args_1)output_ch5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch5_layer_call_and_return_conditional_losses_1605992$
"output_Ch5/StatefulPartitionedCallЛ
"output_Ch4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch4_statefulpartitionedcall_args_1)output_ch4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch4_layer_call_and_return_conditional_losses_1606222$
"output_Ch4/StatefulPartitionedCallЛ
"output_Ch3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch3_statefulpartitionedcall_args_1)output_ch3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch3_layer_call_and_return_conditional_losses_1606452$
"output_Ch3/StatefulPartitionedCallЛ
"output_Ch2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch2_statefulpartitionedcall_args_1)output_ch2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch2_layer_call_and_return_conditional_losses_1606682$
"output_Ch2/StatefulPartitionedCallЛ
"output_Ch1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch1_statefulpartitionedcall_args_1)output_ch1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch1_layer_call_and_return_conditional_losses_1606912$
"output_Ch1/StatefulPartitionedCall╠
!output_Zh/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(output_zh_statefulpartitionedcall_args_1(output_zh_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_output_Zh_layer_call_and_return_conditional_losses_1607142#
!output_Zh/StatefulPartitionedCallу
IdentityIdentity*output_Zh/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

IdentityВ

Identity_1Identity+output_Ch1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1В

Identity_2Identity+output_Ch2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2В

Identity_3Identity+output_Ch3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3В

Identity_4Identity+output_Ch4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4В

Identity_5Identity+output_Ch5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5В

Identity_6Identity+output_Ch6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2H
"output_Ch1/StatefulPartitionedCall"output_Ch1/StatefulPartitionedCall2H
"output_Ch2/StatefulPartitionedCall"output_Ch2/StatefulPartitionedCall2H
"output_Ch3/StatefulPartitionedCall"output_Ch3/StatefulPartitionedCall2H
"output_Ch4/StatefulPartitionedCall"output_Ch4/StatefulPartitionedCall2H
"output_Ch5/StatefulPartitionedCall"output_Ch5/StatefulPartitionedCall2H
"output_Ch6/StatefulPartitionedCall"output_Ch6/StatefulPartitionedCall2F
!output_Zh/StatefulPartitionedCall!output_Zh/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
╬
L
0__inference_max_pooling2d_2_layer_call_fn_160459

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1604532
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ч
г
+__inference_output_Ch2_layer_call_fn_161405

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch2_layer_call_and_return_conditional_losses_1606682
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
м	
я
E__inference_output_Zh_layer_call_and_return_conditional_losses_161362

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
М	
▀
F__inference_output_Ch5_layer_call_and_return_conditional_losses_160599

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         A2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         A2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ч
c
*__inference_dropout_1_layer_call_fn_161335

inputs
identityѕбStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605332
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╩
J
.__inference_max_pooling2d_layer_call_fn_160393

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1603872
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ч
г
+__inference_output_Ch3_layer_call_fn_161423

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch3_layer_call_and_return_conditional_losses_1606452
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Хk
Ѕ
A__inference_model_layer_call_and_return_conditional_losses_160828

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2-
)output_ch6_statefulpartitionedcall_args_1-
)output_ch6_statefulpartitionedcall_args_2-
)output_ch5_statefulpartitionedcall_args_1-
)output_ch5_statefulpartitionedcall_args_2-
)output_ch4_statefulpartitionedcall_args_1-
)output_ch4_statefulpartitionedcall_args_2-
)output_ch3_statefulpartitionedcall_args_1-
)output_ch3_statefulpartitionedcall_args_2-
)output_ch2_statefulpartitionedcall_args_1-
)output_ch2_statefulpartitionedcall_args_2-
)output_ch1_statefulpartitionedcall_args_1-
)output_ch1_statefulpartitionedcall_args_2,
(output_zh_statefulpartitionedcall_args_1,
(output_zh_statefulpartitionedcall_args_2
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб"output_Ch1/StatefulPartitionedCallб"output_Ch2/StatefulPartitionedCallб"output_Ch3/StatefulPartitionedCallб"output_Ch4/StatefulPartitionedCallб"output_Ch5/StatefulPartitionedCallб"output_Ch6/StatefulPartitionedCallб!output_Zh/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1603732 
conv2d/StatefulPartitionedCallщ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1603872
max_pooling2d/PartitionedCall■
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1604872!
dropout/StatefulPartitionedCallО
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1604062"
 conv2d_1/StatefulPartitionedCallЂ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1604202!
max_pooling2d_1/PartitionedCallО
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1604392"
 conv2d_2/StatefulPartitionedCallЂ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1604532!
max_pooling2d_2/PartitionedCallе
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1605332#
!dropout_1/StatefulPartitionedCallС
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:         ђђ*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1605572
flatten/PartitionedCallЛ
"output_Ch6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch6_statefulpartitionedcall_args_1)output_ch6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch6_layer_call_and_return_conditional_losses_1605762$
"output_Ch6/StatefulPartitionedCallЛ
"output_Ch5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch5_statefulpartitionedcall_args_1)output_ch5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch5_layer_call_and_return_conditional_losses_1605992$
"output_Ch5/StatefulPartitionedCallЛ
"output_Ch4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch4_statefulpartitionedcall_args_1)output_ch4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch4_layer_call_and_return_conditional_losses_1606222$
"output_Ch4/StatefulPartitionedCallЛ
"output_Ch3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch3_statefulpartitionedcall_args_1)output_ch3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch3_layer_call_and_return_conditional_losses_1606452$
"output_Ch3/StatefulPartitionedCallЛ
"output_Ch2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch2_statefulpartitionedcall_args_1)output_ch2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch2_layer_call_and_return_conditional_losses_1606682$
"output_Ch2/StatefulPartitionedCallЛ
"output_Ch1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0)output_ch1_statefulpartitionedcall_args_1)output_ch1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_output_Ch1_layer_call_and_return_conditional_losses_1606912$
"output_Ch1/StatefulPartitionedCall╠
!output_Zh/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(output_zh_statefulpartitionedcall_args_1(output_zh_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         A*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_output_Zh_layer_call_and_return_conditional_losses_1607142#
!output_Zh/StatefulPartitionedCallГ
IdentityIdentity*output_Zh/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity▓

Identity_1Identity+output_Ch1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_1▓

Identity_2Identity+output_Ch2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_2▓

Identity_3Identity+output_Ch3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_3▓

Identity_4Identity+output_Ch4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_4▓

Identity_5Identity+output_Ch5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_5▓

Identity_6Identity+output_Ch6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^output_Ch1/StatefulPartitionedCall#^output_Ch2/StatefulPartitionedCall#^output_Ch3/StatefulPartitionedCall#^output_Ch4/StatefulPartitionedCall#^output_Ch5/StatefulPartitionedCall#^output_Ch6/StatefulPartitionedCall"^output_Zh/StatefulPartitionedCall*
T0*'
_output_shapes
:         A2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*ђ
_input_shapeso
m:         ђђ::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"output_Ch1/StatefulPartitionedCall"output_Ch1/StatefulPartitionedCall2H
"output_Ch2/StatefulPartitionedCall"output_Ch2/StatefulPartitionedCall2H
"output_Ch3/StatefulPartitionedCall"output_Ch3/StatefulPartitionedCall2H
"output_Ch4/StatefulPartitionedCall"output_Ch4/StatefulPartitionedCall2H
"output_Ch5/StatefulPartitionedCall"output_Ch5/StatefulPartitionedCall2H
"output_Ch6/StatefulPartitionedCall"output_Ch6/StatefulPartitionedCall2F
!output_Zh/StatefulPartitionedCall!output_Zh/StatefulPartitionedCall:& "
 
_user_specified_nameinputs"»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▓
serving_defaultъ
A
input8
serving_default_input:0         ђђ>

output_Ch10
StatefulPartitionedCall:0         A>

output_Ch20
StatefulPartitionedCall:1         A>

output_Ch30
StatefulPartitionedCall:2         A>

output_Ch40
StatefulPartitionedCall:3         A>

output_Ch50
StatefulPartitionedCall:4         A>

output_Ch60
StatefulPartitionedCall:5         A=
	output_Zh0
StatefulPartitionedCall:6         Atensorflow/serving/predict:╩┤
э~
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
▒__call__
▓_default_save_signature
+│&call_and_return_all_conditional_losses"┴y
_tf_keras_modelДy{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Zh", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Zh", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch1", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch2", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch2", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch3", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch3", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch4", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch4", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch5", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch5", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch6", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch6", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output_Zh", 0, 0], ["output_Ch1", 0, 0], ["output_Ch2", 0, 0], ["output_Ch3", 0, 0], ["output_Ch4", 0, 0], ["output_Ch5", 0, 0], ["output_Ch6", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Zh", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Zh", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch1", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch2", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch2", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch3", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch3", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch4", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch4", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch5", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch5", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_Ch6", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_Ch6", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output_Zh", 0, 0], ["output_Ch1", 0, 0], ["output_Ch2", 0, 0], ["output_Ch3", 0, 0], ["output_Ch4", 0, 0], ["output_Ch5", 0, 0], ["output_Ch6", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "loss", "from_logits": false}}, "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Г"ф
_tf_keras_input_layerі{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 128, 128, 3], "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
»

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerЬ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
ч
	variables
regularization_losses
 trainable_variables
!	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"Ж
_tf_keras_layerл{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
«
"	variables
#regularization_losses
$trainable_variables
%	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"Ю
_tf_keras_layerЃ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
┤

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"Ї
_tf_keras_layerз{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
 
,	variables
-regularization_losses
.trainable_variables
/	keras_api
╝__call__
+й&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
┤

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses"Ї
_tf_keras_layerз{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
 
6	variables
7regularization_losses
8trainable_variables
9	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
▓
:	variables
;regularization_losses
<trainable_variables
=	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"А
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
«
>	variables
?regularization_losses
@trainable_variables
A	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"Ю
_tf_keras_layerЃ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
§

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
к__call__
+К&call_and_return_all_conditional_losses"о
_tf_keras_layer╝{"class_name": "Dense", "name": "output_Zh", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Zh", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch1", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch2", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
╠__call__
+═&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch3", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

Zkernel
[bias
\	variables
]regularization_losses
^trainable_variables
_	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch4", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
л__call__
+Л&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch5", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
 

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
м__call__
+М&call_and_return_all_conditional_losses"п
_tf_keras_layerЙ{"class_name": "Dense", "name": "output_Ch6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch6", "trainable": true, "dtype": "float32", "units": 65, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}}}
с
liter

mbeta_1

nbeta_2
	odecay
plearning_ratemЅmі&mІ'mї0mЇ1mјBmЈCmљHmЉImњNmЊOmћTmЋUmќZmЌ[mў`mЎamџfmЏgmюvЮvъ&vЪ'vа0vА1vбBvБCvцHvЦIvдNvДOvеTvЕUvфZvФ[vг`vГav«fv»gv░"
	optimizer
Х
0
1
&2
'3
04
15
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15
`16
a17
f18
g19"
trackable_list_wrapper
 "
trackable_list_wrapper
Х
0
1
&2
'3
04
15
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15
`16
a17
f18
g19"
trackable_list_wrapper
╗

qlayers
	variables
regularization_losses
rnon_trainable_variables
slayer_regularization_losses
tmetrics
trainable_variables
▒__call__
▓_default_save_signature
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
-
нserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ю

ulayers
	variables
regularization_losses
trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю

ylayers
	variables
regularization_losses
 trainable_variables
znon_trainable_variables
{layer_regularization_losses
|metrics
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ъ

}layers
"	variables
#regularization_losses
$trainable_variables
~non_trainable_variables
layer_regularization_losses
ђmetrics
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
А
Ђlayers
(	variables
)regularization_losses
*trainable_variables
ѓnon_trainable_variables
 Ѓlayer_regularization_losses
ёmetrics
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Ёlayers
,	variables
-regularization_losses
.trainable_variables
єnon_trainable_variables
 Єlayer_regularization_losses
ѕmetrics
╝__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
А
Ѕlayers
2	variables
3regularization_losses
4trainable_variables
іnon_trainable_variables
 Іlayer_regularization_losses
їmetrics
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Їlayers
6	variables
7regularization_losses
8trainable_variables
јnon_trainable_variables
 Јlayer_regularization_losses
љmetrics
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Љlayers
:	variables
;regularization_losses
<trainable_variables
њnon_trainable_variables
 Њlayer_regularization_losses
ћmetrics
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Ћlayers
>	variables
?regularization_losses
@trainable_variables
ќnon_trainable_variables
 Ќlayer_regularization_losses
ўmetrics
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
$:"
ђђA2output_Zh/kernel
:A2output_Zh/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
А
Ўlayers
D	variables
Eregularization_losses
Ftrainable_variables
џnon_trainable_variables
 Џlayer_regularization_losses
юmetrics
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch1/kernel
:A2output_Ch1/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
А
Юlayers
J	variables
Kregularization_losses
Ltrainable_variables
ъnon_trainable_variables
 Ъlayer_regularization_losses
аmetrics
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch2/kernel
:A2output_Ch2/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
А
Аlayers
P	variables
Qregularization_losses
Rtrainable_variables
бnon_trainable_variables
 Бlayer_regularization_losses
цmetrics
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch3/kernel
:A2output_Ch3/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
А
Цlayers
V	variables
Wregularization_losses
Xtrainable_variables
дnon_trainable_variables
 Дlayer_regularization_losses
еmetrics
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch4/kernel
:A2output_Ch4/bias
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
А
Еlayers
\	variables
]regularization_losses
^trainable_variables
фnon_trainable_variables
 Фlayer_regularization_losses
гmetrics
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch5/kernel
:A2output_Ch5/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
А
Гlayers
b	variables
cregularization_losses
dtrainable_variables
«non_trainable_variables
 »layer_regularization_losses
░metrics
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
%:#
ђђA2output_Ch6/kernel
:A2output_Ch6/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
А
▒layers
h	variables
iregularization_losses
jtrainable_variables
▓non_trainable_variables
 │layer_regularization_losses
┤metrics
м__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ъ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
х0
Х1
и2
И3
╣4
║5
╗6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и

╝total

йcount
Й
_fn_kwargs
┐	variables
└regularization_losses
┴trainable_variables
┬	keras_api
Н__call__
+о&call_and_return_all_conditional_losses"щ
_tf_keras_layer▀{"class_name": "MeanMetricWrapper", "name": "output_Zh_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Zh_accuracy", "dtype": "float32"}}
╣

├total

─count
┼
_fn_kwargs
к	variables
Кregularization_losses
╚trainable_variables
╔	keras_api
О__call__
+п&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch1_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch1_accuracy", "dtype": "float32"}}
╣

╩total

╦count
╠
_fn_kwargs
═	variables
╬regularization_losses
¤trainable_variables
л	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch2_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch2_accuracy", "dtype": "float32"}}
╣

Лtotal

мcount
М
_fn_kwargs
н	variables
Нregularization_losses
оtrainable_variables
О	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch3_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch3_accuracy", "dtype": "float32"}}
╣

пtotal

┘count
┌
_fn_kwargs
█	variables
▄regularization_losses
Пtrainable_variables
я	keras_api
П__call__
+я&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch4_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch4_accuracy", "dtype": "float32"}}
╣

▀total

Яcount
р
_fn_kwargs
Р	variables
сregularization_losses
Сtrainable_variables
т	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch5_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch5_accuracy", "dtype": "float32"}}
╣

Тtotal

уcount
У
_fn_kwargs
ж	variables
Жregularization_losses
вtrainable_variables
В	keras_api
р__call__
+Р&call_and_return_all_conditional_losses"ч
_tf_keras_layerр{"class_name": "MeanMetricWrapper", "name": "output_Ch6_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_Ch6_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╝0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
ьlayers
┐	variables
└regularization_losses
┴trainable_variables
Ьnon_trainable_variables
 №layer_regularization_losses
­metrics
Н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
├0
─1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
ыlayers
к	variables
Кregularization_losses
╚trainable_variables
Ыnon_trainable_variables
 зlayer_regularization_losses
Зmetrics
О__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╩0
╦1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
шlayers
═	variables
╬regularization_losses
¤trainable_variables
Шnon_trainable_variables
 эlayer_regularization_losses
Эmetrics
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
щlayers
н	variables
Нregularization_losses
оtrainable_variables
Щnon_trainable_variables
 чlayer_regularization_losses
Чmetrics
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
п0
┘1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
§layers
█	variables
▄regularization_losses
Пtrainable_variables
■non_trainable_variables
  layer_regularization_losses
ђmetrics
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
▀0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Ђlayers
Р	variables
сregularization_losses
Сtrainable_variables
ѓnon_trainable_variables
 Ѓlayer_regularization_losses
ёmetrics
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Т0
у1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Ёlayers
ж	variables
Жregularization_losses
вtrainable_variables
єnon_trainable_variables
 Єlayer_regularization_losses
ѕmetrics
р__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
╝0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
├0
─1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╩0
╦1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
п0
┘1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
▀0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Т0
у1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:, 2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
):'
ђђA2Adam/output_Zh/kernel/m
!:A2Adam/output_Zh/bias/m
*:(
ђђA2Adam/output_Ch1/kernel/m
": A2Adam/output_Ch1/bias/m
*:(
ђђA2Adam/output_Ch2/kernel/m
": A2Adam/output_Ch2/bias/m
*:(
ђђA2Adam/output_Ch3/kernel/m
": A2Adam/output_Ch3/bias/m
*:(
ђђA2Adam/output_Ch4/kernel/m
": A2Adam/output_Ch4/bias/m
*:(
ђђA2Adam/output_Ch5/kernel/m
": A2Adam/output_Ch5/bias/m
*:(
ђђA2Adam/output_Ch6/kernel/m
": A2Adam/output_Ch6/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:, 2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
):'
ђђA2Adam/output_Zh/kernel/v
!:A2Adam/output_Zh/bias/v
*:(
ђђA2Adam/output_Ch1/kernel/v
": A2Adam/output_Ch1/bias/v
*:(
ђђA2Adam/output_Ch2/kernel/v
": A2Adam/output_Ch2/bias/v
*:(
ђђA2Adam/output_Ch3/kernel/v
": A2Adam/output_Ch3/bias/v
*:(
ђђA2Adam/output_Ch4/kernel/v
": A2Adam/output_Ch4/bias/v
*:(
ђђA2Adam/output_Ch5/kernel/v
": A2Adam/output_Ch5/bias/v
*:(
ђђA2Adam/output_Ch6/kernel/v
": A2Adam/output_Ch6/bias/v
Т2с
&__inference_model_layer_call_fn_160946
&__inference_model_layer_call_fn_161233
&__inference_model_layer_call_fn_161270
&__inference_model_layer_call_fn_160863└
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
у2С
!__inference__wrapped_model_160360Й
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
annotationsф *.б+
)і&
input         ђђ
м2¤
A__inference_model_layer_call_and_return_conditional_losses_160779
A__inference_model_layer_call_and_return_conditional_losses_160733
A__inference_model_layer_call_and_return_conditional_losses_161196
A__inference_model_layer_call_and_return_conditional_losses_161109└
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
є2Ѓ
'__inference_conv2d_layer_call_fn_160381О
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
annotationsф *7б4
2і/+                           
А2ъ
B__inference_conv2d_layer_call_and_return_conditional_losses_160373О
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
annotationsф *7б4
2і/+                           
ќ2Њ
.__inference_max_pooling2d_layer_call_fn_160393Я
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
annotationsф *@б=
;і84                                    
▒2«
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_160387Я
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
annotationsф *@б=
;і84                                    
ј2І
(__inference_dropout_layer_call_fn_161300
(__inference_dropout_layer_call_fn_161305┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
─2┴
C__inference_dropout_layer_call_and_return_conditional_losses_161295
C__inference_dropout_layer_call_and_return_conditional_losses_161290┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѕ2Ё
)__inference_conv2d_1_layer_call_fn_160414О
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
annotationsф *7б4
2і/+                           
Б2а
D__inference_conv2d_1_layer_call_and_return_conditional_losses_160406О
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
annotationsф *7б4
2і/+                           
ў2Ћ
0__inference_max_pooling2d_1_layer_call_fn_160426Я
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
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_160420Я
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
annotationsф *@б=
;і84                                    
ѕ2Ё
)__inference_conv2d_2_layer_call_fn_160447О
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
annotationsф *7б4
2і/+                            
Б2а
D__inference_conv2d_2_layer_call_and_return_conditional_losses_160439О
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
annotationsф *7б4
2і/+                            
ў2Ћ
0__inference_max_pooling2d_2_layer_call_fn_160459Я
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
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_160453Я
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
annotationsф *@б=
;і84                                    
њ2Ј
*__inference_dropout_1_layer_call_fn_161335
*__inference_dropout_1_layer_call_fn_161340┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╚2┼
E__inference_dropout_1_layer_call_and_return_conditional_losses_161325
E__inference_dropout_1_layer_call_and_return_conditional_losses_161330┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
(__inference_flatten_layer_call_fn_161351б
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
ь2Ж
C__inference_flatten_layer_call_and_return_conditional_losses_161346б
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
н2Л
*__inference_output_Zh_layer_call_fn_161369б
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
№2В
E__inference_output_Zh_layer_call_and_return_conditional_losses_161362б
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
Н2м
+__inference_output_Ch1_layer_call_fn_161387б
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
F__inference_output_Ch1_layer_call_and_return_conditional_losses_161380б
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
Н2м
+__inference_output_Ch2_layer_call_fn_161405б
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
F__inference_output_Ch2_layer_call_and_return_conditional_losses_161398б
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
Н2м
+__inference_output_Ch3_layer_call_fn_161423б
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
F__inference_output_Ch3_layer_call_and_return_conditional_losses_161416б
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
Н2м
+__inference_output_Ch4_layer_call_fn_161441б
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
F__inference_output_Ch4_layer_call_and_return_conditional_losses_161434б
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
Н2м
+__inference_output_Ch5_layer_call_fn_161459б
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
F__inference_output_Ch5_layer_call_and_return_conditional_losses_161452б
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
Н2м
+__inference_output_Ch6_layer_call_fn_161477б
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
F__inference_output_Ch6_layer_call_and_return_conditional_losses_161470б
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
1B/
$__inference_signature_wrapper_160992input
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 у
!__inference__wrapped_model_160360┴&'01fg`aZ[TUNOHIBC8б5
.б+
)і&
input         ђђ
ф "ЬфЖ
2

output_Ch1$і!

output_Ch1         A
2

output_Ch2$і!

output_Ch2         A
2

output_Ch3$і!

output_Ch3         A
2

output_Ch4$і!

output_Ch4         A
2

output_Ch5$і!

output_Ch5         A
2

output_Ch6$і!

output_Ch6         A
0
	output_Zh#і 
	output_Zh         A┘
D__inference_conv2d_1_layer_call_and_return_conditional_losses_160406љ&'IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                            
џ ▒
)__inference_conv2d_1_layer_call_fn_160414Ѓ&'IбF
?б<
:і7
inputs+                           
ф "2і/+                            ┘
D__inference_conv2d_2_layer_call_and_return_conditional_losses_160439љ01IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                           @
џ ▒
)__inference_conv2d_2_layer_call_fn_160447Ѓ01IбF
?б<
:і7
inputs+                            
ф "2і/+                           @О
B__inference_conv2d_layer_call_and_return_conditional_losses_160373љIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ »
'__inference_conv2d_layer_call_fn_160381ЃIбF
?б<
:і7
inputs+                           
ф "2і/+                           х
E__inference_dropout_1_layer_call_and_return_conditional_losses_161325l;б8
1б.
(і%
inputs         @
p
ф "-б*
#і 
0         @
џ х
E__inference_dropout_1_layer_call_and_return_conditional_losses_161330l;б8
1б.
(і%
inputs         @
p 
ф "-б*
#і 
0         @
џ Ї
*__inference_dropout_1_layer_call_fn_161335_;б8
1б.
(і%
inputs         @
p
ф " і         @Ї
*__inference_dropout_1_layer_call_fn_161340_;б8
1б.
(і%
inputs         @
p 
ф " і         @│
C__inference_dropout_layer_call_and_return_conditional_losses_161290l;б8
1б.
(і%
inputs         @@
p
ф "-б*
#і 
0         @@
џ │
C__inference_dropout_layer_call_and_return_conditional_losses_161295l;б8
1б.
(і%
inputs         @@
p 
ф "-б*
#і 
0         @@
џ І
(__inference_dropout_layer_call_fn_161300_;б8
1б.
(і%
inputs         @@
p
ф " і         @@І
(__inference_dropout_layer_call_fn_161305_;б8
1б.
(і%
inputs         @@
p 
ф " і         @@Е
C__inference_flatten_layer_call_and_return_conditional_losses_161346b7б4
-б*
(і%
inputs         @
ф "'б$
і
0         ђђ
џ Ђ
(__inference_flatten_layer_call_fn_161351U7б4
-б*
(і%
inputs         @
ф "і         ђђЬ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_160420ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_1_layer_call_fn_160426ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_160453ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_2_layer_call_fn_160459ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_160387ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_layer_call_fn_160393ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    і
A__inference_model_layer_call_and_return_conditional_losses_160733─&'01fg`aZ[TUNOHIBC@б=
6б3
)і&
input         ђђ
p

 
ф "жбт
Пџ┘
і
0/0         A
і
0/1         A
і
0/2         A
і
0/3         A
і
0/4         A
і
0/5         A
і
0/6         A
џ і
A__inference_model_layer_call_and_return_conditional_losses_160779─&'01fg`aZ[TUNOHIBC@б=
6б3
)і&
input         ђђ
p 

 
ф "жбт
Пџ┘
і
0/0         A
і
0/1         A
і
0/2         A
і
0/3         A
і
0/4         A
і
0/5         A
і
0/6         A
џ І
A__inference_model_layer_call_and_return_conditional_losses_161109┼&'01fg`aZ[TUNOHIBCAб>
7б4
*і'
inputs         ђђ
p

 
ф "жбт
Пџ┘
і
0/0         A
і
0/1         A
і
0/2         A
і
0/3         A
і
0/4         A
і
0/5         A
і
0/6         A
џ І
A__inference_model_layer_call_and_return_conditional_losses_161196┼&'01fg`aZ[TUNOHIBCAб>
7б4
*і'
inputs         ђђ
p 

 
ф "жбт
Пџ┘
і
0/0         A
і
0/1         A
і
0/2         A
і
0/3         A
і
0/4         A
і
0/5         A
і
0/6         A
џ Н
&__inference_model_layer_call_fn_160863ф&'01fg`aZ[TUNOHIBC@б=
6б3
)і&
input         ђђ
p

 
ф "¤џ╦
і
0         A
і
1         A
і
2         A
і
3         A
і
4         A
і
5         A
і
6         AН
&__inference_model_layer_call_fn_160946ф&'01fg`aZ[TUNOHIBC@б=
6б3
)і&
input         ђђ
p 

 
ф "¤џ╦
і
0         A
і
1         A
і
2         A
і
3         A
і
4         A
і
5         A
і
6         Aо
&__inference_model_layer_call_fn_161233Ф&'01fg`aZ[TUNOHIBCAб>
7б4
*і'
inputs         ђђ
p

 
ф "¤џ╦
і
0         A
і
1         A
і
2         A
і
3         A
і
4         A
і
5         A
і
6         Aо
&__inference_model_layer_call_fn_161270Ф&'01fg`aZ[TUNOHIBCAб>
7б4
*і'
inputs         ђђ
p 

 
ф "¤џ╦
і
0         A
і
1         A
і
2         A
і
3         A
і
4         A
і
5         A
і
6         Aе
F__inference_output_Ch1_layer_call_and_return_conditional_losses_161380^HI1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch1_layer_call_fn_161387QHI1б.
'б$
"і
inputs         ђђ
ф "і         Aе
F__inference_output_Ch2_layer_call_and_return_conditional_losses_161398^NO1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch2_layer_call_fn_161405QNO1б.
'б$
"і
inputs         ђђ
ф "і         Aе
F__inference_output_Ch3_layer_call_and_return_conditional_losses_161416^TU1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch3_layer_call_fn_161423QTU1б.
'б$
"і
inputs         ђђ
ф "і         Aе
F__inference_output_Ch4_layer_call_and_return_conditional_losses_161434^Z[1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch4_layer_call_fn_161441QZ[1б.
'б$
"і
inputs         ђђ
ф "і         Aе
F__inference_output_Ch5_layer_call_and_return_conditional_losses_161452^`a1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch5_layer_call_fn_161459Q`a1б.
'б$
"і
inputs         ђђ
ф "і         Aе
F__inference_output_Ch6_layer_call_and_return_conditional_losses_161470^fg1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ ђ
+__inference_output_Ch6_layer_call_fn_161477Qfg1б.
'б$
"і
inputs         ђђ
ф "і         AД
E__inference_output_Zh_layer_call_and_return_conditional_losses_161362^BC1б.
'б$
"і
inputs         ђђ
ф "%б"
і
0         A
џ 
*__inference_output_Zh_layer_call_fn_161369QBC1б.
'б$
"і
inputs         ђђ
ф "і         Aз
$__inference_signature_wrapper_160992╩&'01fg`aZ[TUNOHIBCAб>
б 
7ф4
2
input)і&
input         ђђ"ЬфЖ
2

output_Ch1$і!

output_Ch1         A
2

output_Ch2$і!

output_Ch2         A
2

output_Ch3$і!

output_Ch3         A
2

output_Ch4$і!

output_Ch4         A
2

output_Ch5$і!

output_Ch5         A
2

output_Ch6$і!

output_Ch6         A
0
	output_Zh#і 
	output_Zh         A