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

ml.math.Vector.prototype.add = function(otherVector) {
	return new ml.math.Vector(
			_.chain(this.values)
				.zip(otherVector.values)
				.map(function(elements) {
					return elements[0] + elements[1];
				})
				.value()
		);
};

ml.math.Vector.prototype.dotProduct = function(otherVector) {
	return _.chain(this.values)
		.zip(otherVector.values)
		.reduce(function(accumulator, numbers) {
			return accumulator + numbers[0] * numbers[1];
		}, 0)
		.value();
};

ml.math.Vector.prototype.print = function() {
	console.log(this.values);
	
	return this;
};

/*
 * Matrix
 */
ml.math.Matrix = function(values) {
	this.values = values;
	this.numberOfRows = values.length;
	this.numberOfColumns = values[0].length;
};

ml.math.Matrix.prototype.transpose = function() {
	return new ml.math.Matrix(_.zip.apply(_, this.values));
};

ml.math.Matrix.prototype.scalarProduct = function(scalar) {
	return new ml.math.Matrix(_.map(this.values, function(row) {
		return _.map(row, function(number) {
			return scalar * number;
		});
	}));
};

ml.math.Matrix.prototype.add = function(otherMatrix) {
	return new ml.math.Matrix(
		_.chain(this.values)
			.zip(otherMatrix.values)
			.map(function(rows) {
				return _.chain(rows[0])
					.zip(rows[1])
					.map(function(elements) {
						return elements[0] + elements[1];
					})
					.value();
			})
			.value()
	);
};

ml.math.Matrix.prototype.mulMatrix = function(otherMatrix) {
	
	// zero-initialized matrix
	var resultValues = [];
	for(var i = 0; i < this.numberOfRows; i++) {
		var row = [];
		for(var j = 0; j < otherMatrix.numberOfColumns; j++) {
			row.push(0);
		}
		resultValues.push(row);
	}

	for(var i = 0; i < this.numberOfRows; i++)
    	for(var j = 0; j < otherMatrix.numberOfColumns; j++)
        	for(var k = 0; k < this.numberOfColumns; k++)
        		resultValues[i][j] += this.values[i][k] * otherMatrix.values[k][j];

	return new ml.math.Matrix(resultValues);
};

ml.math.Matrix.prototype.print = function() {
	console.group('Matrix');
	_.each(this.values, function(row) {
		console.log(row);
	});
	console.groupEnd();
};

