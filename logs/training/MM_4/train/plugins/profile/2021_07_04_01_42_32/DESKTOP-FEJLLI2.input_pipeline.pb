	?H??Q?9@?H??Q?9@!?H??Q?9@      ??!       "?
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
	?u???e???u???e??!?u???e??      ??!       "	=)??+@=)??+@!=)??+@*      ??!       2	l??F????l??F????!l??F????:	+?%@+?%@!+?%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??\p?G@yB???>J@