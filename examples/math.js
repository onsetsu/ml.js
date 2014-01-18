/*
 * sample data
 */
var vec = new ml.math.Vector([4, 3]);
var vec2 = new ml.math.Vector([6,-1]);

console.log(vec.length());
vec.scalarProduct(2).print();
console.log(vec.dotProduct(vec2));

var mat = new ml.math.Matrix([
		[1,2,3],
		[4,5,6]
]);

var mat2 = new ml.math.Matrix([
		[1,1],
		[2,3],
		[4,5]
]);

mat.transpose().print();
mat.scalarProduct(3.5).print();
mat.add(mat).print();
mat.mulMatrix(mat2).print();
mat2.mulMatrix(mat).print();
