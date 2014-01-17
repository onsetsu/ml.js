/*
 * ---------------------------------------------------------------------------
 * models
 * ---------------------------------------------------------------------------
 */

var squaredEuclideanDistance = function(input1, input2) {
	var sum = 0;
	for(var i = 0; i < input1.length; i++)
		sum += Math.pow(input1[i] - input2[i], 2);
	return sum;
};

var euclideanDistance = function(input1, input2) {
	return Math.sqrt(squaredEuclideanDistance(input1, input2));
};

/*
 * k-nearest neighbour classifier
 */
ml.kNearestNeighbour = function(k, trainingsSet) {
	this.k = k;
	this.trainingsSet = trainingsSet;
};

ml.kNearestNeighbour.prototype.classify = function(input) {
	
};

/*
 * Clustering for instances representing real-valued vector.
 */
ml.Clustering = function() {
	
};


