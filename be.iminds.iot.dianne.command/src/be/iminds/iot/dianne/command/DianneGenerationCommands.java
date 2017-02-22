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
package be.iminds.iot.dianne.command;

import java.io.PrintWriter;
import java.util.Collections;

import org.apache.felix.service.command.Descriptor;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(service = Object.class, 
	property = { 
		"osgi.command.scope=dianne",
		"osgi.command.function=generate",
		"osgi.command.function=generateLevel"}, 
	immediate = true)
public class DianneGenerationCommands {

	private Dianne dianne;
	private DiannePlatform platform;

	@Descriptor("Generate a string sequence with a neural network")
	public void generate(
			@Descriptor("neural network to use for generation")
			String nnName, 
			@Descriptor("start string to feed to the neural net first")
			String start, 
			@Descriptor("length of the string to generate")
			int n,
			@Descriptor("optional tags of the neural net to load")
			String... tags) {
		// forward of a rnn
		NeuralNetworkInstanceDTO nni = null;
		try {
			nni = platform.deployNeuralNetwork(nnName, "test rnn", tags);
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
			
			int yDim = 14;
			int xDim = n / yDim;
			
			char[][] output = new char[yDim][xDim]; 

			int y = 0;
			int x = 0;
			boolean down = false;
			
			for(int i = 0; i < start.length(); i++) {
				if(down) {
					output[y][x] = start.charAt((x * yDim) + y);					
				} else {
					output[yDim - y -1][x] = start.charAt((x * yDim) + y);					
				}
				if(y == (yDim - 1)) {
					y = 0;
					x++;
					down = !down;
				} else {
					y++;
				}
			}
					
			for (int i = 0; i < start.length() - 1; i++) {
				nextChar(nn, start.charAt(i));
			}

			char c = start.charAt(start.length() - 1);
			for (int i = start.length(); i < n; i++) {
				c = nextChar(nn, c);
				if(down) {
					output[y][x] = c;					
				} else {
					output[yDim - y -1][x] = c;					
				}
				if(y == (yDim - 1)) {
					y = 0;
					x++;
					down = !down;
				} else {
					y++;
				}
			}
			
			PrintWriter writer = new PrintWriter("output.txt");
			
			for(int i = 0; i < yDim; i++) {
				writer.println(new String(output[i]));
			}
			writer.close();

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			platform.undeployNeuralNetwork(nni);
		}
	}
	
	@Descriptor("Generate a level with a neural network")
	public void generateLevel(
			@Descriptor("neural network to use for generation")
			String nnName, 
			@Descriptor("start level to feed to the neural net first")
			String start, 
			@Descriptor("length of the level to generate")
			int n,
			@Descriptor("difficulty of the level to generate")
			int difficulty,
			@Descriptor("optional tags of the neural net to load")
			String... tags) {
		// forward of a rnn
		NeuralNetworkInstanceDTO nni = null;
		try {
			nni = platform.deployNeuralNetwork(nnName, "test rnn", tags);
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
			
			int yDim = 14;
			int xDim = n / yDim;
			
			char[][] output = new char[yDim][xDim]; 

			int y = 0;
			int x = 0;
			boolean down = false;
			
			for(int i = 0; i < start.length(); i++) {
				if(down) {
					output[y][x] = start.charAt((x * yDim) + y);					
				} else {
					output[yDim - y -1][x] = start.charAt((x * yDim) + y);					
				}
				if(y == (yDim - 1)) {
					y = 0;
					x++;
					down = !down;
				} else {
					y++;
				}
			}
					
			for (int i = 0; i < start.length() - 1; i++) {
				nextChar(nn, start.charAt(i), difficulty);
			}

			char c = start.charAt(start.length() - 1);
			for (int i = start.length(); i < n; i++) {
				c = nextChar(nn, c, difficulty);
				if(down) {
					output[y][x] = c;					
				} else {
					output[yDim - y -1][x] = c;					
				}
				if(y == (yDim - 1)) {
					y = 0;
					x++;
					down = !down;
				} else {
					y++;
				}
			}
			
			PrintWriter writer = new PrintWriter("output.txt");
			
			for(int i = 0; i < yDim; i++) {
				writer.println(new String(output[i]));
			}
			writer.close();

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			platform.undeployNeuralNetwork(nni);
		}
	}

	private char nextChar(NeuralNetwork nn, char current) {
		// construct input tensor
		String[] labels = nn.getOutputLabels();
		if (labels == null) {
			throw new RuntimeException(
					"Neural network " + nn.getNeuralNetworkInstance().name + " is not trained and has no labels");
		}
		Tensor in = new Tensor(labels.length);
		in.fill(0.0f);
		int index = 0;
		for (int i = 0; i < labels.length; i++) {
			if (labels[i].charAt(0) == current) {
				index = i;
				break;
			}
		}
		in.set(1.0f, index);

		// forward
		Tensor out = nn.forward(in);

		// select next, sampling from (Log)Softmax output
		if (TensorOps.min(out) < 0) {
			// assume logsoftmax output, take exp
			out = TensorOps.exp(out, out);
		}
		
		int m = 0;
		for(int i = 1; i < out.size(); i++) {
			if(out.get(m) < out.get(i)) {
				m = i;
			}
		}
				
		double s = 0, r = Math.random();
		int o = 0;
		while (o < out.size() && (s += out.get(o)) < r) {
			o++;
		}
		
		//Sample character with highest probability instead of random sample if probability of said sample is lower than 10%
		if(out.get(o) < 0.1) {
			o = m;
		}
		
		return labels[o].charAt(0);
	}
	
	private char nextChar(NeuralNetwork nn, char current, int difficulty) {
		// construct input tensor
		String[] labels = nn.getOutputLabels();
		if (labels == null) {
			throw new RuntimeException(
					"Neural network " + nn.getNeuralNetworkInstance().name + " is not trained and has no labels");
		}
		Tensor in = new Tensor(labels.length + 3);
		in.fill(0.0f);
		int index = 0;
		for (int i = 0; i < labels.length; i++) {
			if (labels[i].charAt(0) == current) {
				index = i;
				break;
			}
		}
		in.set(1.0f, index);
		in.set(1.0f, labels.length + difficulty);

		// forward
		Tensor out = nn.forward(in);

		// select next, sampling from (Log)Softmax output
		if (TensorOps.min(out) < 0) {
			// assume logsoftmax output, take exp
			out = TensorOps.exp(out, out);
		}
		
		int m = 0;
		for(int i = 1; i < labels.length; i++) {
			if(out.get(m) < out.get(i)) {
				m = i;
			}
		}
				
		double s = 0, r = Math.random();
		int o = 0;
		while (o < labels.length && (s += out.get(o)) < r) {
			o++;
		}
		
		//Sample character with highest probability instead of random sample if probability of said sample is lower than 10%
		if(out.get(o) < 0.1) {
			o = m;
		}
		
		return labels[o].charAt(0);
	}

	@Reference
	void setDianne(Dianne d) {
		this.dianne = d;
	}

	@Reference
	void setDiannePlatform(DiannePlatform p) {
		this.platform = p;
	}

}
