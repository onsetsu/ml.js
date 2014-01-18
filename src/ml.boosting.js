/*
 * ---------------------------------------------------------------------------
 * models
 * ---------------------------------------------------------------------------
 */

/*
 * Enhanced Basic Linear Classifier.
 */

ml.EnhancedBasicLinearClassifier = function() {};

ml.EnhancedBasicLinearClassifier.prototype.learn = function(trainingSet) {
	var numberOfPositives = 0;
	var numberOfNegatives = 0;
	var instances = trainingSet.getInstances();

	var p = new ml.math.Vector(_.range(instances[0].input.length).map(function () { return 0; }));
	var n = new ml.math.Vector(_.range(instances[0].input.length).map(function () { return 0; }));
	
	_.each(instances, function(instance) {
		if(instance.output == 1) {
			p = p.add(new ml.math.Vector(instance.input));
			numberOfPositives++;
		} else {
			n = n.add(new ml.math.Vector(instance.input));
			numberOfNegatives++;
		}
	});
	p = p.scalarProduct(1/numberOfPositives);
	n = n.scalarProduct(1/numberOfNegatives);

	p.print();
	n.print();
	
	this.w = p.add(n.scalarProduct(-1)).print();
	this.t = (Math.pow(p.length(), 2) - Math.pow(n.length(), 2)) / 2;
	console.log(this.t);
	return this;
};

// 1 if w*x > t
// 0 else
ml.EnhancedBasicLinearClassifier.prototype.classify = function(x) {
	return sign(this.w.dotProduct(new ml.math.Vector(x)) - this.t);
};

ml.EnsembleModel = function() {
	
};

ml.Boosting = function() {
	
};
