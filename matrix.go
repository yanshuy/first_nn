package main

type matrix [][]float64

func Transpose(mat matrix) matrix {
	m := len(mat)
	n := len(mat[0])

	t := make([][]float64, n)
	for i := range n {
		t[i] = make([]float64, m)
	}

	for i := range n {
		for j := range m {
			t[i][j] = mat[j][i]
		}
	}
	return t
}

func (mat matrix) transform(transformFunc func(x float64, i int, j int) float64) matrix {
	m := len(mat)
	n := len(mat[0])

	newMat := make(matrix, m)
	for i := range m {
		r := make([]float64, n)
		newMat[i] = r
		for j := range n {
			newMat[i][j] = transformFunc(mat[i][j], i, j)
		}
	}

	return newMat
}

func (mat1 matrix) Mul(mats ...matrix) matrix{
	mat1M := len(mat1)
	mat1N := len(mat1[0])

	newMat := make(matrix, mat1M)
	for i := range mat1M {
		r := make([]float64, mat1N)
		newMat[i] = r
		for j := range mat1N {
			newMat[i][j] = mat1[i][j]
			for _, mat2 := range mats {
				newMat[i][j] *= mat2[i][j]
			}
		}
	}
	return newMat
}

// dot product of matrix
func MulMat(mat1 matrix, mat2 matrix) matrix {
	mat1M := len(mat1)
	mat1N := len(mat1[0])
	mat2M := len(mat2)
	mat2N := len(mat2[0])
	
	if mat1N != mat2M {
		panic("multiplication not possible")
	}

	result := make([][]float64, mat1M)
	for i := range result {
		result[i] = make([]float64, mat2N)
	}

	for i := range mat1M {
		for j := range mat2N {
			sum := 0.0
			for k := range mat1N {
				sum += mat1[i][k] * mat2[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

