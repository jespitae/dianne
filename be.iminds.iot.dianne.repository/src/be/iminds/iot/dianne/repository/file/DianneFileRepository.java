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
package be.iminds.iot.dianne.repository.file;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(immediate=true)
public class DianneFileRepository implements DianneRepository {

	private String dir = "nn";
	
	private Map<RepositoryListener, List<String>> listeners = Collections.synchronizedMap(new HashMap<RepositoryListener, List<String>>());
	protected ExecutorService executor = Executors.newSingleThreadExecutor();

	
	@Activate
	public void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.storage");
		if(s!=null){
			dir = s;
		}
		File d = new File(dir+"/weights/");
		d.mkdirs();
	}
	
	@Deactivate
	public void deactivate(){
		executor.shutdownNow();
	}

	@Override
	public List<ModuleTypeDTO> availableCompositeModules(){
		List<ModuleTypeDTO> composites = new ArrayList<ModuleTypeDTO>();
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(!f.isDirectory())
				continue;

			// a composite module is identified by a .composite configuration file in the nn folder
			File configFile = new File(f.getPath()+File.separator+".composite");
			if(!configFile.exists())
				continue;
			
			String type = f.getName();
			String category = "Composite";
			
			// the .composite file can contain a number of configurable properties of the component
			// these are formatted one per line, with on each line
			// <property name>,<property key>,<property class type, i.e. java.lang.Integer>
			// these properties can be referred to in the Neural Network description as ${<property key>} 
			try {
				List<String> lines = Files.readAllLines(configFile.toPath());
				List<ModulePropertyDTO> properties = lines.stream().map(line -> {
					String[] entries = line.split(",");
					return new ModulePropertyDTO(entries[0], entries[1], entries[2]);
				}).collect(Collectors.toList());
				ModulePropertyDTO[] array = new ModulePropertyDTO[properties.size()];
				properties.toArray(array);
	
				ModuleTypeDTO compositeType = new ModuleTypeDTO(type, category, true, array);
				composites.add(compositeType);
			} catch(Exception e ){
				e.printStackTrace();
			}
		}
		return composites;
	}
	
	@Override
	public List<String> availableNeuralNetworks() {
		List<String> nns = new ArrayList<String>();
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(!f.isDirectory())
				continue;
				
			String name = f.getName();
			if(!name.equals("weights")){
				nns.add(f.getName());
			}
		}
		return nns;
	}

	@Override
	public NeuralNetworkDTO loadNeuralNetwork(String nnName){
		try {
			String nn = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/modules.txt")));
			return DianneJSONConverter.parseJSON(nn);
		} catch (IOException e) {
			throw new RuntimeException("Failed to load neural network "+nnName, e);
		}
	}
	
	@Override
	public void storeNeuralNetwork(NeuralNetworkDTO nn){
		File d = new File(dir+"/"+nn.name);
		d.mkdirs();
		
		File n = new File(dir+"/"+nn.name+"/modules.txt");
		
		File locked = new File(dir+"/"+nn.name+"/.locked");
		if(locked.exists()){
			throw new RuntimeException("Cannot store neural network "+nn.name+", this is locked!");
		}
		
		String output = DianneJSONConverter.toJsonString(nn, true);
		
		try(PrintWriter p = new PrintWriter(n)) {
			p.write(output);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public String loadLayout(String nnName) throws IOException {
		String layout = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/layout.txt")));
		return layout;
	}
	
	@Override
	public void storeLayout(String nnName, String layout){
		File l = new File(dir+"/"+nnName+"/layout.txt");

		try(PrintWriter p = new PrintWriter(l)) {
			p.write(layout);
		} catch(Exception e){
			e.printStackTrace();
		}		
	}
	
	@Override
	public synchronized Tensor loadParameters(UUID moduleId, String... tag) {
		return load(moduleId, tag);
	}

	@Override
	public synchronized Map<UUID, Tensor> loadParameters(Collection<UUID> moduleIds,
			String... tag) {
		return moduleIds.stream().collect(
				Collectors.toMap(moduleId -> moduleId, moduleId -> loadParameters(moduleId, tag)));
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String nnName, String... tag) throws Exception {
		Map<UUID, Tensor> parameters  = new HashMap<>();
		
		NeuralNetworkDTO nn = loadNeuralNetwork(nnName);
		// TODO should we deduce based on ModuleDTO whether the module is trainable and throw
		// exception when trainable module has no parameters on file system?
		for(ModuleDTO m : nn.modules.values()){
			try {
				parameters.put(m.id, load(m.id, tag));
			} catch(Exception e){
				// ignore if no parameters found for a module
			}
		}
		if(parameters.isEmpty())
			throw new Exception("No parameters available for NN "+nnName);
		return parameters;
	}
	
	private Tensor load(UUID moduleId, String... tag){
		try {
			// first check weights, next check all other nn dirs
			File f = new File(dir+"/weights/"+parametersId(moduleId, tag));
			if(!f.exists()){
				f = null;
				File d = new File(dir);
				for(String l : d.list()){
					f = new File(dir+"/"+l+"/"+parametersId(moduleId, tag));
					if(f.exists()){
						break;
					} else {
						f = null;
					}
				}
			}
			if(f!=null){
				// load tensor in chuncks, slightly slower than one copy from Java to native,
				// but reduces memory usage a lot for big tensors
				int bufferSize = 10000;
				float[] data = new float[bufferSize];
				DataInputStream is = new DataInputStream(new BufferedInputStream(new FileInputStream(f)));
				int length = is.readInt();
				Tensor t = new Tensor(length);
				int index = 0;
				while(length > 0){
					if(length<bufferSize){
						bufferSize = length;
						data = new float[bufferSize];
					}
					for(int i=0;i<bufferSize;i++){
						data[i] = is.readFloat();
					}
					
					t.narrow(0, index, bufferSize).set(data);;
					
					length -= bufferSize;
					index+= bufferSize;
				}
				is.close();
				return t;
			}
			throw new FileNotFoundException();
		} catch(Exception e){
			throw new RuntimeException("Failed to load parameters for module "+moduleId+" with tags "+Arrays.toString(tag), e);
		}
	}
	
	@Override
	public synchronized void storeParameters(UUID nnId, UUID moduleId, Tensor parameters, String... tag) {
		store(moduleId, parameters, tag);
		
		notifyListeners(nnId, Collections.singleton(moduleId), tag);
	}
	
	@Override
	public synchronized void storeParameters(UUID nnId, Map<UUID, Tensor> parameters, String... tag) {
		parameters.entrySet().stream().forEach(e -> store(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(parameters.keySet());
		notifyListeners(nnId, uuids, tag);
	}
	
	@Override
	public synchronized void accParameters(UUID nnId, UUID moduleId, Tensor accParameters, String... tag){
		acc(moduleId, accParameters, tag);
		
		notifyListeners(nnId, Collections.singleton(moduleId), tag);
	}

	@Override
	public synchronized void accParameters(UUID nnId, Map<UUID, Tensor> accParameters, String... tag) {
		accParameters.entrySet().stream().forEach(e -> acc(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(accParameters.keySet());
		notifyListeners(nnId, uuids, tag);

	}
	
	@Override
	public long spaceLeft() {
		File d = new File(dir);
		return d.getUsableSpace();
	}
	
	private void store(UUID moduleId, Tensor parameters, String... tag){
		File f = new File(dir+"/weights/"+parametersId(moduleId, tag));

		try(DataOutputStream os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
			float[] data = parameters.get();
			os.writeInt(data.length);
			for(int i=0;i<data.length;i++){
				os.writeFloat(data[i]);
			}
			os.flush();
			os.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private void acc(UUID moduleId, Tensor accParameters, String... tag){
		Tensor parameters = accParameters;
		try {
			parameters = load(moduleId, tag);
			
			TensorOps.add(parameters, parameters, accParameters);
		} catch(Exception e){
			System.out.println("Failed to load parameters for "+moduleId+" "+Arrays.toString(tag)+", store as new");
		}
	
		store(moduleId, parameters, tag);
	}
	
	private String parametersId(UUID id, String[] tag){
		String pid = id.toString();
		if(tag!=null && tag.length>0){
			for(String t : tag){
				if(t!=null)
					pid+="-"+t;
			}
		}
		return pid;
	}
	
	private void notifyListeners(UUID nnId, Collection<UUID> moduleIds, String... tag){
		synchronized(listeners){
			// match tags and nnId
			List<String> tags = new ArrayList<String>(tag.length+1);
			tags.addAll(Arrays.asList(tag));
			tags.add(nnId.toString());
			final List<RepositoryListener> toNotify = listeners.entrySet()
					.stream()
					.filter( e -> match(e.getValue(), moduleIds, tags))
					.map( e -> e.getKey())
					.collect(Collectors.toList());
			
			executor.execute( ()->{
				for(RepositoryListener l : toNotify){
					l.onParametersUpdate(nnId, moduleIds, tag);
				}
			});
			
		}
	}
	
	private boolean match(Collection<String> targets, Collection<UUID> moduleIds, List<String> tags){
		// match everything if targets = null
		if(targets==null){
			return true;
		}
		
		// targets in form  moduleId:tag
		for(String target : targets){
			String[] split = target.split(":");
			if(split[0].length()!=0){
				// moduleId provided
				if(!moduleIds.contains(UUID.fromString(split[0]))){
					return false;
				}
			}
			
			// some tag provided
			for(int i=1;i<split.length;i++){
				String t = split[i];
				if(tags!=null){
					if(!tags.contains(t)){
						return false;
					}
				} else {
					return false;
				}
			}
			return true;
		}
		return false;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addRepositoryListener(RepositoryListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			listeners.put(l, Arrays.asList(targets));
		} else {
			listeners.put(l, null);
		}
	}
	
	void removeRepositoryListener(RepositoryListener l){
		listeners.remove(l);
	}

}
