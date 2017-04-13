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
package be.iminds.iot.dianne.rnn.learn.strategy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.CriterionConfig;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.strategy.GenerativeAdverserialLearnProgress;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingStrategy;
import be.iminds.iot.dianne.rnn.learn.strategy.config.GenerativeAdverserialSequenceConfig;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Generative Adversarial NN Learning strategy
 * 
 * Based on https://gist.github.com/lopezpaz/16a67948325fc97239775b09b261677a
 * 
 * @author tverbele
 *
 */
@SuppressWarnings("rawtypes")
public class WassersteinGenerativeAdverserialSequenceLearningStrategy implements LearningStrategy {

	protected SequenceDataset dataset;
	
	protected NeuralNetwork generator;
	protected NeuralNetwork discriminator;
	
	protected GenerativeAdverserialSequenceConfig config;
	
	protected GradientProcessor gradientProcessorG;
	protected GradientProcessor gradientProcessorD;

	protected Criterion criterion;
	
	protected SequenceSamplingStrategy sampling;
	
	protected Map<UUID, List<Tensor>> memories;
	protected List<Tensor> inputs;
	
	protected Sequence<Batch> sequence = null;
		
	protected Tensor target;
	protected Tensor output;
	protected Tensor gumbelSample;
	
	protected float temperature;
	protected float annealStep;
	protected float epsilon;
	protected int n;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = (SequenceDataset) dataset;
		this.temperature = 5;
		this.n = 5;
		this.epsilon = (float) (1f * Math.pow(10,-20));
		this.annealStep = 0.0004f;

		this.generator = nns[0];
		this.discriminator = nns[1];
		
		String[] labels = this.dataset.getLabels();
		if(labels!=null)
			this.generator.setOutputLabels(labels);		
		
		this.config = DianneConfigHandler.getConfig(config, GenerativeAdverserialSequenceConfig.class);
			
		sampling = SequenceSamplingFactory.createSamplingStrategy(this.config.sampling, this.dataset, config);
		criterion = CriterionFactory.createCriterion(CriterionConfig.BCE, config);
		gradientProcessorG = ProcessorFactory.createGradientProcessor(this.config.method, generator, config);
		gradientProcessorD = ProcessorFactory.createGradientProcessor(this.config.method, discriminator, config);
		
		target = new Tensor(this.config.batchSize, 1);
		gumbelSample = new Tensor(this.config.batchSize, this.config.generatorDim);
		
		// Initialize lists to store inputs and memories
		memories = new HashMap<>();
		for(UUID id : generator.getMemories().keySet()) {
			memories.put(id, new ArrayList<>());
		}
		inputs = new ArrayList<>();
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		//Anneal temperature
		if(temperature > 1) {
			temperature = temperature - 0.0004f;
		}	
		
		// Clear delta params
		generator.zeroDeltaParameters();
		discriminator.zeroDeltaParameters();
		
		// Reset memory
		discriminator.resetMemory(config.batchSize);
		generator.resetMemory(config.batchSize);
		
		// Sample sequences and indexes
		int[] sequences = sampling.sequence(config.batchSize);
		int[] indexes = sampling.next(sequences, config.sequenceLength);

		// These should be classified as correct by the discriminator
		target.fill(0.85f);
	
		// Load minibatch of real data for the discriminator 
		sequence = dataset.getBatchedSequence(sequence , sequences, indexes, config.sequenceLength);
				
		output = discriminator.forward(sequence.getInputs()).get(config.sequenceLength - 1);
		float d_loss_positive = TensorOps.mean(criterion.loss(output, target));
		Tensor gradOutput = criterion.grad(output, target);
		discriminator.backward(gradOutput);
				
		// Keep gradients to the parameters
		discriminator.accGradParameters();	
				
		// Reset memory discriminator
		discriminator.resetMemory(config.batchSize);
		
		// These should be classified as incorrect by discriminator
		target.fill(0.15f);
		
		// Generate sequence of data
		generateSequence();
						
		output = discriminator.forward(sequence.getTargets()).get(config.sequenceLength - 1);
		float d_loss_negative = TensorOps.mean(criterion.loss(output, target));
		gradOutput = criterion.grad(output, target);
		discriminator.backward(gradOutput);	
				
		// Keep gradients to the parameters
		discriminator.accGradParameters();		
		
		// Run gradient processors
		gradientProcessorD.calculateDelta(i);
			
		discriminator.updateParameters();
		
		for(UUID id : discriminator.getParameters().keySet()) {
			Tensor params = discriminator.getParameters().get(id);
			TensorOps.clamp(params, params, -0.01f, 0.01f);
		}

		float g_loss = 0;
		
		if(n == 1) {
			g_loss = trainGenerator(i);
			n = 5;
		} else {
			n--;
		}		
	
		return new GenerativeAdverserialLearnProgress(i, d_loss_positive, d_loss_negative, g_loss);
	}

	private void generateSequence() {		
		// Save memory of network
		for(UUID id : generator.getMemories().keySet()) {
			Tensor memory = generator.getMemories().get(id).getMemory();
			if(memories.get(id).isEmpty()) {
				memories.get(id).add(new Tensor(memory.dims()));				
			}
			memories.get(id).get(0).set(memory.get());
		}		
		
		Batch start = sequence.get(0); 
		start.getInput().randn();
		Tensor out = generator.forward(start.getInput());
		start.getTarget().set(gumbelSoftmax(out, 0).get());
		
		for(int s = 1; s < config.sequenceLength; s++) {
			// Save memory of network
			for(UUID id : generator.getMemories().keySet()) {
				Tensor memory = generator.getMemories().get(id).getMemory();
				if(memories.get(id).size() <= s) {
					memories.get(id).add(new Tensor(memory.dims()));
				}
				memories.get(id).get(s).set(memory.get());
			}
			
			Batch batch = sequence.get(s);
			batch.getInput().set(sequence.get(s - 1).getTarget().get());
			out = generator.forward(batch.getInput());				
			batch.getTarget().set(gumbelSoftmax(out, s).get());
		}
	}
		
	//Create gumbel-softmax sample
	private Tensor gumbelSoftmax(Tensor tensor, int s) {
		// Create new gumbel sample
		gumbelSample();
		
		TensorOps.add(tensor, tensor, gumbelSample);
		TensorOps.div(tensor, tensor, temperature);
		if(inputs.size() <= s) {
			inputs.add(new Tensor(tensor.dims()));
		}
		inputs.get(s).set(tensor.get());
		
		ModuleOps.softmax(tensor, tensor);
		return tensor;
	}
	
	//Calculate gumbel sample
	private void gumbelSample() {
		gumbelSample.rand();	
		TensorOps.add(gumbelSample, gumbelSample, epsilon);
		TensorOps.log(gumbelSample, gumbelSample);
		TensorOps.mul(gumbelSample, gumbelSample, -1f);
		TensorOps.add(gumbelSample, gumbelSample, epsilon);
		TensorOps.log(gumbelSample, gumbelSample);
		TensorOps.mul(gumbelSample, gumbelSample, -1f);
	}
	
	private float trainGenerator(long i) {
		// Reset memory discriminator
		discriminator.resetMemory(config.batchSize);
		
		// This should be classified correct by discriminator to get the gradient improving the generator
		target.fill(0.85f);
					
		output = discriminator.forward(sequence.getTargets()).get(config.sequenceLength - 1);			
		float g_loss = TensorOps.mean(criterion.loss(output, target));
		Tensor gradOutput = criterion.grad(output, target);
		Tensor gradInput = discriminator.backward(gradOutput);
			
		//Differentiate gradients
		ModuleOps.softmaxGradIn(gradInput, gradInput, inputs.get(config.sequenceLength - 1), sequence.get(config.sequenceLength - 1).getTarget());
		TensorOps.div(gradInput, gradInput, temperature);
		
		gradInput = generator.backward(gradInput);
			
		// Keep gradients to the parameters
		generator.accGradParameters();
							
		for(int s = config.sequenceLength - 2; s >= 0; s--) {
			//Set memory
			for(UUID id : memories.keySet()) {
				generator.getMemory(id).setMemory(memories.get(id).get(s));
			}
			//Forward input
			generator.forward(sequence.get(s).getInput());
			
			//Differentiate gradients
			ModuleOps.softmaxGradIn(gradInput, gradInput, inputs.get(s), sequence.get(s).getTarget());
			TensorOps.div(gradInput, gradInput, temperature);
				
			gradInput = generator.backward(gradInput);

			// Keep gradients to the parameters
			generator.accGradParameters();			
		}
				
		// Run gradient processors
		gradientProcessorG.calculateDelta(i);
		
		// Update parameters
		generator.updateParameters();
		
		return g_loss;
	}
}
