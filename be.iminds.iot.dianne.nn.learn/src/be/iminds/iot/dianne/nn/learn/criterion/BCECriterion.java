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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class BCECriterion implements Criterion {
	
	protected static final float EPS = 1e-12f;
	
	protected Tensor output;
	protected Tensor target;

	protected Tensor invOut;
	protected Tensor invTar;
	
	protected Tensor epsOut;
	protected Tensor epsInvOut;
	
	protected Tensor logOut;
	protected Tensor logInvOut;
	
	protected Tensor l;
	
	protected Tensor loss = new Tensor(1);
	protected Tensor grad;
	
	protected BatchConfig b;
	
	public BCECriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public Tensor loss(Tensor output, Tensor target) {
		this.output = output;
		this.target = target;
		
		invOut = TensorOps.mul(invOut, output, -1);
		TensorOps.add(invOut, invOut, 1);
		invTar = TensorOps.mul(invTar, target, -1);
		TensorOps.add(invTar, invTar, 1);
		
		epsOut = TensorOps.add(epsOut, output, EPS);
		epsInvOut = TensorOps.add(epsInvOut, invOut, EPS);
		
		logOut = TensorOps.log(logOut, epsOut);
		logInvOut = TensorOps.log(logInvOut, epsInvOut);
		
		l = TensorOps.cmul(l, target, logOut);
		TensorOps.addcmul(l, l, 1, invTar, logInvOut);
		if(b.batchSize > 1){
			loss.reshape(b.batchSize);
			for(int i=0;i<b.batchSize;i++){
				loss.set(-TensorOps.sum(l.select(0, i)), i);
			}
		} else {
			loss.set(-TensorOps.sum(l), 0);
		}
		
		return loss;
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		if(!output.equals(this.output) || !target.equals(this.target)) {
			invOut = TensorOps.mul(invOut, output, -1);
			TensorOps.add(invOut, invOut, 1);
			
			epsOut = TensorOps.add(epsOut, output, EPS);
			epsInvOut = TensorOps.add(epsInvOut, invOut, EPS);
		}
		
		grad = TensorOps.sub(grad, output, target);
		
		TensorOps.cdiv(grad, grad, epsOut);
		TensorOps.cdiv(grad, grad, epsInvOut);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
		
		return grad;
	}

}
