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
package be.iminds.iot.dianne.api.nn.learn;

/**
 * Strategy to select a next index to visit in the learning procedure.
 * 
 * @author tverbele
 *
 */
public interface SamplingStrategy {

	/**
	 * @return next index of the dataset to visit
	 */
	int next();
	
	/**
	 * @param count number of indices to generate
	 * @return count indices of the dataset to visit
	 */
	default int[] next(int count){
		int[] indices = new int[count];
		for(int i=0;i<count;i++){
			indices[i] = next();
		}
		return indices;
	}
}
