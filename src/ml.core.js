// define namespace
var ml = {};

/*
 * Feature
 * 
 * Describes a feature type.
 */
ml.Feature = function(name) {
	this.name = name;
};

ml.Feature.prototype.setDomain = function(domain) {
	this.domain = domain;
	
	return this;
};

ml.Feature.Categorical = 0;
ml.Feature.Continuous = 1;

ml.Feature.prototype.setType = function(type) {
	this.type = type;
	
	return this;
};

/*
 * Instance
 * 
 * represents a single instance (of the trainingSet)
 */
ml.Instance = function(instance, label) {
	this.input = instance;
	this.output = label;
};

/*
 * TrainingSet
 */
ml.TrainingSet = function() {
	this.instances = [];
};

ml.TrainingSet.prototype.add = function(instance, label) {
	this.instances.push(new ml.Instance(instance, label));
};

ml.TrainingSet.prototype.getInstances = function() {
	return this.instances;
};

ml.TrainingSet.prototype.setInstances = function(instances) {
	this.instances = instances;
	
	return this;
};

ml.TrainingSet.prototype.getNumberOfInstances = function() {
	return this.instances.length;
};

// TRAINING SET UTILS

// Split the TrainingSet into multiple TrainingSets using the discrete values of given feature.
ml.TrainingSet.prototype.splitIntoSubsets = function(featureIndex) {
	var rawSplit = _.groupBy(this.instances, function(instance) {
		return instance.input[featureIndex];
	});
	
	// wrap into TrainingSets
	for(var i in rawSplit) {
		rawSplit[i] = new ml.TrainingSet().setInstances(rawSplit[i]);
	}
	
	return rawSplit;
};

// Returns the majority class of the TrainingSet.
ml.TrainingSet.prototype.getMajorityClass = function() {
	return _.chain(this.getInstances())
		.countBy("output")
		.pairs()
		.max(_.last)
		.head()
		.value();
};

/*
 * Utility functions
 */
var sign = function(number) {
	if(number > 0) return 1;
	if(number < 0) return -1;
	return 0;
};

