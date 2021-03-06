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
package be.iminds.iot.dianne.api.dataset;

import java.util.ArrayList;
import java.util.List;

/**
 * A helper class for representing raw data of sequence of samples/batches of a dataset.
 * 
 * This can be used to tansfer dataset data from a remote dataset 
 * without creating precious native tensors all the time.
 * 
 * @author tverbele
 *
 */
public class RawBatchedSequence {

	public List<? extends RawBatch> data;
	
	public RawBatchedSequence(List<? extends RawBatch> data){
		this.data = data;
	}

	public Sequence<? extends Batch> copyInto(Sequence<? extends Batch> b){
		if(b == null){
			List<Batch> d = new ArrayList<>();
			for(RawBatch rs : data){
				d.add(rs.copyInto(null));
			}
			return new Sequence<Batch>(d);
		} else {
			for(int i=0;i<data.size();i++){
				data.get(i).copyInto(b.data.get(i));
			}
			b.size = data.size();
			return b;
		}
	}
}