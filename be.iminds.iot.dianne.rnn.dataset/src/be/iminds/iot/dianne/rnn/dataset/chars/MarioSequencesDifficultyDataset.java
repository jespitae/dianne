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
package be.iminds.iot.dianne.rnn.dataset.chars;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * A Character SequenceDataset ... read a text file as a single sequence
 * 
 * @author tverbele
 *
 */
@Component(
		service={SequenceDataset.class, Dataset.class},
		immediate=true,
		property={"aiolos.unique=true","aiolos.combine=*"},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.MarioSequencesDifficultyDataset")
public class MarioSequencesDifficultyDataset extends AbstractDataset implements SequenceDataset<Sample, Batch> {

	private String[] files = {"mario-1-1.txt", "mario-1-2.txt", "mario-1-3.txt", "mario-2-1.txt", "mario-3-3.txt", 
			"mario-4-1.txt", "mario-4-2.txt", "mario-5-3.txt", "mario-6-1.txt", "mario-6-2.txt", "mario-6-3.txt", "mario-8-1.txt",
			"SuperMarioBros2(J)-World1-1.txt", "SuperMarioBros2(J)-World1-3.txt", "SuperMarioBros2(J)-World2-1.txt"
			, "SuperMarioBros2(J)-World2-2.txt", "SuperMarioBros2(J)-World2-3.txt", "SuperMarioBros2(J)-World3-3.txt"
			, "SuperMarioBros2(J)-World4-1.txt", "SuperMarioBros2(J)-World4-3.txt", "SuperMarioBros2(J)-World5-2.txt"
			, "SuperMarioBros2(J)-World6-3.txt", "SuperMarioBros2(J)-WorldA-3.txt", "SuperMarioBros2(J)-WorldB-3.txt"};
	
	private enum Difficulty {
		LOW, MEDIUM, HIGH;
	}
	
	private Difficulty[] difficulties = {Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, 
			Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW,
			Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW,
			Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW,
			Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW, Difficulty.LOW};
	
	private String[] data;
	private String allData = "";
	private String chars = "";

	@Override
	protected void init(Map<String, Object> properties) {
		try {
			inputType = "character";
			targetType = "characer";
				
			data = new String[files.length * 2];
			noSamples = 0;
			
			for(int t = 0; t < files.length; t++) {

				// read the data
				byte[] encoded = Files.readAllBytes(Paths.get(dir+File.separator+files[t]));
				data[t] = new String(encoded, Charset.defaultCharset());
				data[files.length + t] = new String(encoded, Charset.defaultCharset());
				
				String snakedDataUp = "";
				String snakedDataDown = "";
				int dimY = 0;
				
				for(int i = 0; i < data[t].length(); i++) {
					if(data[t].charAt(i) == '\n') {
						dimY++;
					}
				}
				
				int dimX = (data[t].length() / dimY) - 1;
				
				boolean down = false;
				
				for(int x = 0; x < dimX; x++) {
					for(int y = 0; y < dimY; y++) {
						if(down) {
							snakedDataUp += "" + data[t].charAt((y * (dimX + 1)) + x);
							snakedDataDown += "" + data[files.length + t].charAt(((dimY - (y + 1)) * (dimX + 1)) + x);
						} else {
							snakedDataUp += "" + data[t].charAt(((dimY - (y + 1)) * (dimX + 1)) + x);
							snakedDataDown += "" + data[files.length + t].charAt((y * (dimX + 1)) + x);
						}
					}
					down = !down;
				}
				
				data[t] = snakedDataUp;
				data[files.length + t] = snakedDataDown;
				
				allData += data[t];
				allData += data[files.length + t];
				
				noSamples += (data[t].length() * 2);
				
				// no labels given, build up the vocabulary
				if(!properties.containsKey("labels") 
						&& !properties.containsKey("labelsFile")){
					
					data[t].chars().forEach(c -> {
						if(!chars.contains(""+(char)c)){
							chars+=""+(char)c;
						}
					});
				}
			}
			
			labels = new String[chars.length()];
			for(int i=0;i<labels.length;i++){
				labels[i] = ""+chars.charAt(i);
			}
			
			inputDims = new int[]{chars.length()};
			targetDims = new int[]{chars.length()};
			
		} catch(Exception e){
			e.printStackTrace();
			throw new RuntimeException("Failed to load char sequence dataset", e);
		}	
	}

	@Override
	protected void readLabels(String labelsFile) {
		try {
			byte[] encoded = Files.readAllBytes(Paths.get(dir+File.separator+labelsFile));
			chars = new String(encoded, Charset.defaultCharset());
			
			labels = new String[chars.length()];
			for(int i=0;i<labels.length;i++){
				labels[i] = ""+chars.charAt(i);
			}
			
			inputDims = new int[]{chars.length()};
			targetDims = new int[]{chars.length()};
			
		} catch(Exception e){
			e.printStackTrace();
			throw new RuntimeException("Failed to load char sequence dataset", e);
		}	
	}

	@Override
	protected Tensor getInputSample(Tensor t, int index) {
		return asTensor(allData.charAt(index), t);
	}

	@Override
	protected Tensor getTargetSample(Tensor t, int index) {
		return asTensor(allData.charAt(index+1), t);
	}

	private Tensor asTensor(char c, Tensor t){
		int index = 0;
		index = chars.indexOf(c);
		if(t == null)
			t = new Tensor(chars.length());
		t.fill(0.0f);
		if(index == -1){
			System.err.println("Character "+c+" is not in the vocabulary");
			return t;
		}
		t.set(1.0f, index);
		return t;
	}
	
	private Tensor asTensor(char c, Tensor t, int sequence){
		int index = 0;
		index = chars.indexOf(c);
		if(t == null)
			t = new Tensor(chars.length() + 3);
		t.fill(0.0f);
		if(index == -1) {
			System.err.println("Character "+c+" is not in the vocabulary");
			return t;
		}
		t.set(1.0f, index);
		if(difficulties[sequence] == Difficulty.LOW) {
			t.set(1.0f, chars.length());
		} else { 
			if(difficulties[sequence] == Difficulty.MEDIUM) {
				t.set(1.0f, chars.length() + 1);
			} else {
				t.set(1.0f, chars.length() + 2);
			}
		}
		return t;
	}
	
	private char asChar(Tensor t){
		int index = TensorOps.argmax(t);
		return chars.charAt(index);
	}

	@Override
	public int sequences() {
		return (files.length * 2);
	}
	
	@Override
	public int sequenceLength(int index){
		if(index > sequences()){
			throw new ArrayIndexOutOfBoundsException();
		}
		return data[index].length();
	}

	@Override
	public Sequence<Sample> getSequence(Sequence<Sample> seq, int sequence, int index, int length) {
		if(seq == null){
			seq = new Sequence<Sample>();
		}
		List<Sample> s = seq.data;
		
		if(sequence > sequences()){
			throw new RuntimeException("Invalid sequence number");
		}

		if(index >= data[sequence].length()){
			throw new RuntimeException("Invalid start index: "+index);
		}
		
		if(length == -1){
			length = data[sequence].length();
		}
		
		Sample previous = null;
		for(int i=0;i<length;i++){
			Sample sample;
			if(s.size() <= i){
				sample = new Sample(previous != null ? previous.target : new Tensor(chars.length()), new Tensor(chars.length()));
				s.add(sample);
			} else {
				sample = s.get(i);
				if(previous != null){
					sample.input = previous.target;
				}
			}
			
			int k = index + i;
			if(i == 0){
				char c = data[sequence].charAt(k);
				asTensor(c, sample.input);
			} 
			
			if(k+1 < data[sequence].length()){
				char c = data[sequence].charAt(k+1);
				asTensor(c, sample.target);
			} else {
				sample.target.fill(Float.NaN);
			}
			
			previous = sample;
		}
		
		seq.size = length;
		return seq;
	}

	@Override
	public Sequence<Batch> getBatchedSequence(Sequence<Batch> seq, int[] sequences, int[] indices, int length) {
		if(seq == null){
			seq = new Sequence<Batch>();
		}
		List<Batch> b = seq.data;
		
		for(int sequence : sequences){
			if(sequence > sequences()){
				throw new RuntimeException("Invalid sequence number");
			}
		}

		if(indices != null){
			for(int i = 0; i < indices.length; i++){
				if(indices[i] >= data[sequences[i]].length()){
					throw new RuntimeException("Invalid start index");
				}
			}
		}
		
		if(length == -1){
			length = data[0].length();
			for(int i = 0; i < sequences(); i++){
				if(data[i].length() < length){
					length = data[i].length();
				}
			}
		}
		
		Batch previous = null;
		for(int i=0;i<length;i++){
			Batch batch;
			if(b.size() <= i){
				batch = new Batch(previous != null ? previous.target : new Tensor(sequences.length, chars.length() + 3), new Tensor(sequences.length, chars.length() + 3));
				b.add(batch);
			} else {
				batch = b.get(i);
				if(previous != null){
					batch.input = previous.target;
				}
			}
			
			for(int s=0;s<sequences.length;s++){
				int start = indices == null ? 0 : indices[s];
				int k = start + i;
				
				if(i == 0){
					char c = data[sequences[s]].charAt(k);
					asTensor(c, batch.getSample(s).input);
				} 
				
				if(k+1 < data[sequences[s]].length()){
					char c = data[sequences[s]].charAt(k+1);
					asTensor(c, batch.getSample(s).target);
				} else {
					batch.getSample(s).target.fill(Float.NaN);
				}
			}
			
			previous = batch;
		}
		
		seq.size = length;
		return seq;
	}
}
