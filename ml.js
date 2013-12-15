(function(window) {
	var getStepFunction = function(threshold) {
		return function stepFunction(input) {
			return input > threshold ? 1 : -1;
		};
	};
	
	var getLinearCombinationFunction = function(weights, size) {
		return function linearCombination(instance) {
			var sum = weights[0];
			for(var i = 1; i <= size; i++)
				sum += instance[i-1] * weights[i];
			return sum;
		};
	};
	
	// define namespace
	var ml = {};
	ml.nn = {};
	
	/*
	 * trainingSet
	 */
	ml.TrainingSet = function() {
		this.size = 0;
		this.instances = [];
		this.labels = [];
	};

	ml.TrainingSet.prototype.add = function(instance, label) {
		this.size++;
		this.instances.push(instance);
		this.labels.push(label);
	};
	
	/*
	 * models
	 */
	
	// Simplest ANN: Perceptron
	ml.nn.Perceptron = function(inputSize) {
		this.inputSize = inputSize;

		// default weights
		this.weights = [];
		this.assignRandomWeights();
		
		// default functions
		this.linkFunction = getLinearCombinationFunction(this.weights, this.inputSize);
		this.activationFunction = getStepFunction(0);
	};
	
	// assign randow weights (with w_0 as additional weight)
	ml.nn.Perceptron.prototype.assignRandomWeights = function() {
		for(var i = 0; i <= this.inputSize; i++)
			this.weights[i] = Math.random();
	};
	
	ml.nn.Perceptron.prototype.classify = function(instance) {
		return this.activationFunction(this.linkFunction(instance));
	};

	ml.nn.Perceptron.prototype.originalTraining = function(trainingSet, learningRate) {
		this.assignRandomWeights();
		
		var classifiedSomeInstancesWrong = true;
		
		while(classifiedSomeInstancesWrong) {
			classifiedSomeInstancesWrong = false;
			
			// iterate all instances in the given trainingSet
			for(var i = 0; i < trainingSet.size; i++) {
				// for each weight
				for(var j = 0; j < this.weights.length; j++) {
					// compute delta_w_j
					var delta_w_j = learningRate * (trainingSet.labels[i] - this.classify(trainingSet.instances[i])) * (j == 0 ? 1 : trainingSet.instances[i][j-1]);
					
					// update w_j
					this.weights[j] += delta_w_j;
					
					// found an incorrectly classified instance -> Then take another iteration
					if(delta_w_j != 0) classifiedSomeInstancesWrong = true;
				}
			}
		}
	};
	
	ml.nn.Perceptron.prototype.gradientDescent = function(trainingSet, learningRate, acceptedErrorThreshold) {
		this.assignRandomWeights();
		
		var squaredError = Number.MAX_VALUE;
		
		while(squaredError > acceptedErrorThreshold) {
			// initialize all delta_w_i to zero
			var delta_w = [];
			for(var i = 0; i < this.weights.length; i++) delta_w.push(0);
			
			// iterate all instances in the given trainingSet
			for(var i = 0; i < trainingSet.size; i++) {
				// for each weight
				for(var j = 0; j < this.weights.length; j++) {
					// accumulate delta_w_j (learn update)
					delta_w[j] += learningRate * (trainingSet.labels[i] - this.classify(trainingSet.instances[i])) * (j == 0 ? 1 : trainingSet.instances[i][j-1]);
				}
			}
			
			// update all w_i at the end of an iteration
			for(var i = 0; i < delta_w.length; i++) this.weights[i] += delta_w[i];
			
			// compute new Error using squaredError function
			squaredError = 0;
			for(var i = 0; i < trainingSet.size; i++) {
				squaredError += Math.pow((trainingSet.labels[i] - this.classify(trainingSet.instances[i])), 2);
			}
			squaredError /= 2;
		}
	};
	
	ml.nn.Perceptron.prototype.stochasticGradientDescent = function() {};

	// complex types
	ml.nn.DoubleLayerPerceptron = function() {};
	ml.nn.MultiLayerPerceptron = function(layerSizes) {};
	
	// make globally visible
	window.ml = ml;
})(window);

// sample usage
var trainingSet = new ml.TrainingSet();
trainingSet.add([10,0,0], 1);
trainingSet.add([5,3,-1], 1);
trainingSet.add([8,0,0], 1);
trainingSet.add([6,1,0], 1);
trainingSet.add([0,2,10], -1);
trainingSet.add([1,2,12], -1);
trainingSet.add([0,0,30], -1);

var p = new ml.nn.Perceptron(3);
//p.originalTraining(trainingSet, 0.1);
p.gradientDescent(trainingSet, 0.1, 0.05);

console.log("ready");