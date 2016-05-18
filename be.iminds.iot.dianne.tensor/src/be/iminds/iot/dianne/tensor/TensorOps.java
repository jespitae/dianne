/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.tensor;

/**
 * Provides all supported Tensor operations. Each operation where a tensor is returned,
 * also has the argument res, in which one could provide a tensor in which the result
 * will be put and returned. This in order to save memory allocations. When res is null 
 * a new Tensor object will be created.
 * 
 * @author tverbele
 *
 */
public class TensorOps {

	/**
	 * Add the given value to all elements in the T.
	 */
	public static native Tensor add(Tensor res, final Tensor tensor, final float value);

	/**
	 * Add tensor1 to tensor2 and put result into res. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor add(Tensor res, final Tensor tensor1, final Tensor tensor2);

	/**
	 * Multiply elements of tensor2 by the scalar value and add it to tensor1. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor add(Tensor res, final Tensor tensor1, final float value, final Tensor tensor2);
	
	/**
	 * Subract the given value of all elements in the T.
	 */
	public static native Tensor sub(Tensor res, final Tensor tensor, final float value);

	/**
	 * Subtract tensor2 from tensor1 and put result into res. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor sub(Tensor res, final Tensor tensor1, final Tensor tensor2);
	
	/**
	 * Multiply elements of tensor2 by the scalar value and subtract it from tensor1. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor sub(Tensor res, final Tensor tensor1, final float value, final Tensor tensor2);
	
	/**
	 * Multiply all elements in the tensor by the given value.
	 */
	public static native Tensor mul(Tensor res, final Tensor tensor, final float value);

	/**
	 * Element-wise multiplication of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor cmul(Tensor res, final Tensor tensor1, final Tensor tensor2);

	/**
	 * Divide all elements in the T by the given value.
	 */
	public static native Tensor div(Tensor res, final Tensor tensor1, final float value);

	/**
	 * Element-wise division of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public static native Tensor cdiv(Tensor res, final Tensor tensor1, final Tensor tensor2);
	
	/**
	 * Performs the dot product between vec1 and vec2. 
	 * The number of elements must match: both Ts are seen as a 1D vector.
	 */
	public static native float dot(final Tensor vec1, final Tensor vec2);
	
	/**
	 * Performs the matrix product between vec1 and vec2
	 * @param res placeholder
	 * @param vec1 vector of size m
	 * @param vec2 vector of size n
	 * @return resulting matrix of size mxn
	 */
	public static native Tensor vv(Tensor res, final Tensor vec1, final Tensor vec2);

	/**
	 * Matrix vector product of mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a n x m matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public static native Tensor mv(Tensor res, final Tensor mat, final Tensor vec);
	
	/**
	 * Matrix vector product of transposed mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a m x n matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public static native Tensor tmv(Tensor res, final Tensor mat, final Tensor vec);

	/**
	 * Matrix matrix product of matensor1 and matensor2. If matensor1 is a n x m matrix, matensor2 a m x p matrix, 
	 * res must be a n x p matrix.
	 */
	public static native Tensor mm(Tensor res, final Tensor mat1, final Tensor mat2);
	
	/**
	 * Matrix matrix product of transposed matensor1 and matensor2. If matensor1 is a m x n matrix, matensor2 a m x p matrix, 
	 * res must be a n x p matrix.
	 */
	public static native Tensor tmm(Tensor res, final Tensor mat1, final Tensor mat2);
	
	/**
	 * Performs the matrix product between vec1 and vec2 and adds this to mat
	 * @param res placeholder
	 * @param mat mxn matrix to add to result
	 * @param vec1 vector of size m
	 * @param vec2 vector of size n
	 * @return resulting matrix of size mxn
	 */
	public static native Tensor addvv(Tensor res, final Tensor mat, final Tensor vec1, final Tensor vec2);
 
	/**
	 * Performs a matrix-vector multiplication between mat (2D tensor) and vec (1D tensor) 
	 * and add it to vec1. In other words, res = vec1 + mat*vec2
	 */
	public static native Tensor addmv(Tensor res, final Tensor vec1, final Tensor mat, final Tensor vec2);

	/**
	 * Performs a matrix-vector multiplication between matensor1 (2D tensor) and matensor2 (2D tensor) 
	 * and add it to mat. In other words, res = mat + matensor1*matensor2
	 */
	public static native Tensor addmm(Tensor res, final Tensor mat, final Tensor mat1, final Tensor mat2);
	
	/**
	 * Calculates element-wise exp function
	 */
	public static native Tensor exp(Tensor res, final Tensor tensor);

	/**
	 * Calculates element-wise log function
	 */
	public static native Tensor log(Tensor res, final Tensor tensor);
	
	/**
	 * Calculate sqrt for each element
	 */
	public static native Tensor sqrt(Tensor res, final Tensor tensor);
	
	/**
	 * Return the sum of all elements
	 */
	public static native float sum(final Tensor tensor);
	
	/**
	 * Return the max of all elements
	 */
	public static native float max(final Tensor tensor);
	
	/**
	 * Return the min of all elements
	 */
	public static native float min(final Tensor tensor);
	
	/**
	 * Return the mean of all elements
	 */
	public static native float mean(final Tensor tensor);
	
	/**
	 * Return index of the max element (treats T as 1 dim vector)
	 */
	public static native int argmax(final Tensor tensor);
	
	/**
	 * Return index of the min element (treats T as 1 dim vector)
	 */
	public static native int argmin(final Tensor tensor);

	/**
	 * Scale (bilinear interpollate) in 2 dimensions
	 * In case of 3D tensor it will scale all 'channels'
	 */
	public static native Tensor scale2D(Tensor res, final Tensor t, final int... dims);

	/**
	 * First crop to not stretch the image before scaling
	 */
	public static native Tensor frame(Tensor res, final Tensor t, final int... dims);
	
}

