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

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
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
public class GenerativeAdverserialSequenceLearningStrategy implements LearningStrategy {

	protected SequenceDataset dataset;
	
	protected NeuralNetwork generator;
	protected NeuralNetwork discriminator;
	
	protected GenerativeAdverserialSequenceConfig config;
	
	protected GradientProcessor gradientProcessorG;
	protected GradientProcessor gradientProcessorD;

	protected Criterion criterion;
	
	protected SequenceSamplingStrategy sampling;
	
	protected Sequence<Batch> sequence;
	
	protected Tensor target;
	
	protected int temperature;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = (SequenceDataset) dataset;
		this.temperature = 5;
		
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
		
		target = new Tensor(this.config.batchSize);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		//Anneal temperature
		if(i != 1 && (i % 500) == 0) {
			i--;
		}		
		
		// Clear delta params
		generator.zeroDeltaParameters();
		discriminator.zeroDeltaParameters();
		
		discriminator.resetMemory(config.batchSize);
		
		// sample sequence
		int[] sequences = sampling.sequence(config.batchSize);
		int[] indexes = sampling.next(sequences, config.sequenceLength);
	
		// First update the discriminator
		
		// Load minibatch of real data for the discriminator 
		sequence = dataset.getBatchedSequence(sequence, sequences, indexes, config.sequenceLength);
		
		// These should be classified as correct by discriminator
		target.fill(0.85f);
		target.reshape(config.batchSize, 1);
		Tensor output = discriminator.forward(sequence.getInputs()).get(config.sequenceLength - 1);
		float d_loss_positive = TensorOps.mean(criterion.loss(output, target));
		Tensor gradOutput = criterion.grad(output, target);
		discriminator.backward(gradOutput);
		
		// Keep gradients to the parameters
		discriminator.accGradParameters();
		
		// these should be classified as incorrect by discriminator
		target.fill(0.15f);
		sequence.data.clear();		
		generateSequence();
						
		output = discriminator.forward(sequence.getInputs()).get(config.sequenceLength - 1);
		float d_loss_negative = TensorOps.mean(criterion.loss(output, target));
		gradOutput = criterion.grad(output, target);
		discriminator.backward(gradOutput);
		
		// Update discriminator weights
		discriminator.accGradParameters();
		
		// Run gradient processors
		gradientProcessorD.calculateDelta(i);
		
		discriminator.updateParameters();
		
		// This should be classified correct by discriminator to get the gradient improving the generator
		target.fill(0.85f);
		sequence.data.clear();
		
		generateSequence();
		output = discriminator.forward(sequence.getInputs()).get(config.sequenceLength - 1);
		
		float g_loss = TensorOps.mean(criterion.loss(output, target));
		gradOutput = criterion.grad(output, target);
		Tensor gradInput = discriminator.backward(gradOutput);
		generator.backward(gradInput);
		// Update generator weights
		generator.accGradParameters();
		
		// Run gradient processors
		gradientProcessorG.calculateDelta(i);
		
		// Update parameters
		generator.updateParameters();
		
		return new GenerativeAdverserialLearnProgress(i, d_loss_positive, d_loss_negative, g_loss);
	}

	private void generateSequence() {	
		generator.resetMemory(0);
		Batch start = new Batch(new Tensor(config.batchSize, config.generatorDim), new Tensor(config.batchSize, config.generatorDim));
		for(int b = 0; b < config.batchSize; b++) {
			start.getInput(b).randn();			
			start.getInput(b).set(gumbelSoftmax(start.getInput(b)).get());
			Tensor out = generator.forward(start.getInput(b));			
			start.getTarget(b).set(gumbelSoftmax(out).get());
		}
		sequence.data.add(start);
		for(int s = 1; s < config.sequenceLength; s++) {
			Batch batch = new Batch(new Tensor(config.batchSize, config.generatorDim), new Tensor(config.batchSize, config.generatorDim));
			for(int b = 0; b < config.batchSize; b++) {
				batch.getInput(b).set(sequence.data.get(s - 1).getTarget(b).get());
				Tensor out = generator.forward(batch.getInput(b));				
				batch.getTarget(b).set(gumbelSoftmax(out).get());
			}
			sequence.data.add(batch);
		}
	}
		
	//Create gumbel-softmax sample
	private Tensor gumbelSoftmax(Tensor tensor) {
		Tensor gumbelSample = new Tensor(config.generatorDim);
		gumbelSample.rand();
		float epsilon = (float) (1f * Math.pow(10,-20));
		
		//Calculate gumbel sample
		TensorOps.add(gumbelSample, gumbelSample, epsilon);
		TensorOps.log(gumbelSample, gumbelSample);
		TensorOps.mul(gumbelSample, gumbelSample, -1f);
		TensorOps.add(gumbelSample, gumbelSample, epsilon);
		TensorOps.log(gumbelSample, gumbelSample);
		TensorOps.mul(gumbelSample, gumbelSample, -1f);
		
		//Calculate gumbel-softmax sample
		TensorOps.add(tensor, tensor, gumbelSample);
		TensorOps.div(tensor, tensor, temperature);
		ModuleOps.softmax(tensor, tensor);
		return tensor;
	}
}
