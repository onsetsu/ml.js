ml.math = {};
/*
 * Vector
 */
ml.math.Vector = function(values) {
	this.values = values;
};

ml.math.Vector.prototype.length = function() {
	return Math.sqrt(
		_.reduce(this.values, function(accumulator, value) {
			return accumulator + value * value;
		}, 0));
};

ml.math.Vector.prototype.scalarProduct = function(scalar) {
	return new ml.math.Vector(_.map(this.values, function(value) {
		return scalar * value;
	}));
};

ml.math.Vector.prototype.dotProduct = function(otherVector) {
	return _.chain(this.values)
		.zip(otherVector.values)
		.reduce(function(accumulator, numbers) {
			return accumulator + numbers[0] * numbers[1];
		}, 0)
		.value();
};

