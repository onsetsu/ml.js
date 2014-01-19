/*
 * ---------------------------------------------------------------------------
 * models
 * ---------------------------------------------------------------------------
 */

/*
 * Enhanced Basic Linear Classifier.
 */

ml.EnhancedBasicLinearClassifier = function() {};

ml.EnhancedBasicLinearClassifier.prototype.learn = function(trainingSet, weights) {
	var totalPositiveWeight = 0;
	var totalNegativeWeight = 0;
	var instances = trainingSet.getInstances();

	var p = new ml.math.Vector(_.range(instances[0].input.length).map(function () { return 0; }));
	var n = new ml.math.Vector(_.range(instances[0].input.length).map(function () { return 0; }));
	
	// calculate p (n) as weighted average of all positive (negative) instances
	_.each(instances, function(instance, i) {
		var currentWeight = weights[i];
		if(instance.output == 1) {
			p = p.add(new ml.math.Vector(instance.input).scalarProduct(currentWeight));
			totalPositiveWeight += currentWeight;
		} else {
			n = n.add(new ml.math.Vector(instance.input).scalarProduct(currentWeight ));
			totalNegativeWeight += currentWeight;
		}
	});
	p = p.scalarProduct(1/totalPositiveWeight);
	n = n.scalarProduct(1/totalNegativeWeight);

	this.w = p.add(n.scalarProduct(-1));
	this.t = (Math.pow(p.length(), 2) - Math.pow(n.length(), 2)) / 2;
	
	return this;
};

// 1 if w*x > t
// 0 else
ml.EnhancedBasicLinearClassifier.prototype.classify = function(x) {
	return sign(this.w.dotProduct(new ml.math.Vector(x)) - this.t);
};

/*
 * Ensemble model as set of weighted models.
 */

ml.EnsembleModel = function() {
	this.models = [];
	this.confidenceFactors = [];
};

ml.EnsembleModel.prototype.add = function(model, confidenceFactor) {
	this.models.push(model);
	this.confidenceFactors.push(confidenceFactor);
};

ml.EnsembleModel.prototype.adjustConfidence = function(i, confidenceFactor) {
	this.confidenceFactors[i] = confidenceFactor;
};

ml.EnsembleModel.prototype.classify = function(x) {
	var sum = 0;
	
	for(var i = 0; i < this.models.length; i++)
		sum += this.confidenceFactors[i] * this.models[i].classify(x);
	
	return sign(sum);
};

/*
 * Actual Boosting algorithm
 */
ml.Boosting = function(trainingSet, ensembleSize, Algorithm) {
	var ensemble = new ml.EnsembleModel();
	
	// Initial weights are all 1/|trainingSet|
	var sizeOfTrainingSet = trainingSet.getNumberOfInstances();
	var currentWeights = _.range(sizeOfTrainingSet).map(function () {
		return 1/sizeOfTrainingSet;
	});

	for(var i = 0; i < ensembleSize; i++) {
		// add new model to ensemble with current weights and default confidence factor
		var currentModel = new ml.EnhancedBasicLinearClassifier()
			.learn(trainingSet, currentWeights);
		var confidenceFactor = 1;
		ensemble.add(currentModel, confidenceFactor);
		
		// calculate relative error of current ensemble on training set
		var relativeError = _.reduce(trainingSet.getInstances(), function(accumulator, instance) {
			if(ensemble.classify(instance.input) != instance.output)
				return accumulator + 1;
			return accumulator;
		}, 0) / sizeOfTrainingSet;
		console.log("relative error", relativeError);

		// break if random guess would be better
		// (relative error > 1/2)
		if(relativeError > 1/2) {
			// remove latest model
			ensemble.models.length--;
			ensemble.confidenceFactors.length--;
			return ensemble;
		};
		
		// calculate confidence factor alpha
		// update confidence factor alpha
		confidenceFactor = (1/2) * Math.log((1 - relativeError) / relativeError) / Math.log(Math.E);
		ensemble.adjustConfidence(i, confidenceFactor);
		
		// update weights depending on if given instance was correctly classified
		currentWeights = _.chain(currentWeights)
			.zip(trainingSet.getInstances())
			.map(function(pair, i) {
				var weight = pair[0];
				var instance = pair[1];
				// if misclassified
				if(ensemble.classify(instance.input) != instance.output)
					return weight / (2 * relativeError);
				else
					// if correctly classified
					return weight / (2 * (1 - relativeError));
			})
			.value();
	}
	return ensemble;
};
