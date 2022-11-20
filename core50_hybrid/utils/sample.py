
def train_gwr_fc(model, gwr_model, criterion, optimizer,cnn_optimizer, scheduler, cnn_scheduler, train_loader,
					num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, transform = None):
	model.train()
	# model.eval()
	for epoch in range(1, num_epochs + 1):						#vara
		print('Epoch : ', epoch)
		if task_id != 0:
			for image_names, labels in train_loader:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				fc_labels = labels % num_classes_in_task
				optimizer.zero_grad()
				with torch.set_grad_enabled(True):
					outputs = model(inputs)
					loss = criterion(outputs, fc_labels)
					loss.backward()	#inputs = model.fc.weight, model.fc.bias)
					optimizer.step()
					# cnn_optimizer.step()
			scheduler.step()
			# cnn_scheduler.step()
		else:
			for image_names, labels in train_loader:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				fc_labels = labels % num_classes_in_task
				optimizer.zero_grad()
				with torch.set_grad_enabled(True):
					outputs = model(inputs)
					loss = criterion(outputs, fc_labels)
					loss.backward()
					optimizer.step()
					cnn_optimizer.step()
			scheduler.step()
			cnn_scheduler.step()

	feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
	feature_model.eval()
	optimizer.zero_grad()		# feature extraction
	with torch.set_grad_enabled(False):
		features = torch.tensor([]).to(device)
		gwr_labels = torch.tensor([]).to(device)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			features_batch = feature_model(inputs)

			features_batch = features_batch.view(features_batch.data.shape[0],512)
			# print(features_batch.data.shape)
			gwr_labels_batch = labels // num_classes_in_task
			features = torch.cat((features, features_batch))
			gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
		# print(features.data.shape)
		features = features.cpu()[::gwr_imgs_skip]
		gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]
		# print('features shape: ', features.shape)

		if(task_id == 0):
			gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
		else:
			gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing
		# graph_gwr = gwr_model.train(output, n_epochs=epochs)
		# Xg = gwr_model.get_positions()

		# number_of_clusters1 = nx.number_connected_components(graph_gwr1)	# number of dintinct clusters without any connections
		# print('number of clusters: ',number_of_clusters1)
	num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
	print('number of nodes:', num_nodes1)
		
				
	return model, gwr_model