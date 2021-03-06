?	?H??Q?9@?H??Q?9@!?H??Q?9@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?H??Q?9@?u???e??1=)??+@Al??F????I+?%@rEagerKernelExecute 0*	q=
ף?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?{/?h??!R?g?F@)?} R?8??1?`?j RD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??s??Ƨ?![隬?<@)T??????1B????3@:Preprocessing2U
Iterator::Model::ParallelMapV2?_?+?ۖ?!
v?"?*@)?_?+?ۖ?1
v?"?*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??h:;??!?N???? @)??h:;??1?N???? @:Preprocessing2F
Iterator::Model&?5?Р?!?????3@)Ϡ?????1?7?Cl@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice???????!|?@秧@)???????1|?@秧@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7?W????!??I?
T@)??F!ɬ~?1?-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??4?(??!)!$l??G@)?]???h?1?:?? U??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??\p?G@QB???>J@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?u???e???u???e??!?u???e??      ??!       "	=)??+@=)??+@!=)??+@*      ??!       2	l??F????l??F????!l??F????:	+?%@+?%@!+?%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??\p?G@yB???>J@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop0B?O ??!0B?O ??0"&
CudnnRNNCudnnRNN??b?D??!??????"(

concat_1_0ConcatV2K0?Dv??!??BcS??"(
gradients/AddNAddN?3,????!c6??E???"*
transpose_9	Transpose????????!??v???"C
$gradients/transpose_9_grad/transpose	Transposey?jY{G??!???t+T??""
split_1Split?Ӌ?fo??!??R????"*
transpose_0	Transpose?????[??!?3?.W???"A
"gradients/transpose_grad/transpose	TransposeR?UV<??!2?=????";
gradients/split_1_grad/concatConcatV2?-??a|?!?????N??Q      Y@Y_?V*?@a=K?fX@qC??lX@y?2??XŐ?"?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 