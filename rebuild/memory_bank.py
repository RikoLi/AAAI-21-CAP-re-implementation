import torch
import numpy as np

def _intersection(indices_a, indices_b):
    '''Get intersection of two indice tensors.'''
    intersec = set(indices_a.tolist()) & set(indices_b.tolist())
    return torch.tensor(list(intersec))

class IntraCamLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, percam_proxies, abs_proxy_label, memory, momentum):
        ctx.momentum = momentum
        ctx.percam_proxies = percam_proxies
        ctx.memory = memory
        
        ctx.save_for_backward(input_feat, abs_proxy_label)

        return input_feat.mm(percam_proxies.t())

    @staticmethod
    def backward(ctx, grad_output):
        input_feat, abs_proxy_label = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(ctx.percam_proxies)
        
        # update memory bank
        for x, y in zip(input_feat, abs_proxy_label):
            ctx.memory[y] = ctx.momentum * ctx.memory[y] + (1.0 - ctx.momentum) * x
            ctx.memory[y] /= ctx.memory[y].norm()

        return grad_input, None, None, None, None


class ProxyMemoryBank(torch.nn.Module):
    def __init__(self, features, global_dataset, momentum=0.2, temp=0.07):
        super(ProxyMemoryBank, self).__init__()
        self.features = features
        self.global_dataset = global_dataset
        self.momentum = momentum
        self.temp = temp

        self.mapper, _memory = self._init_memory()
        self.register_buffer('memory', _memory)

    def _init_memory(self):
        feature_dim = self.features.size(1)
        
        # get abs proxy labels
        abs_proxy_labels = np.array([sample[-1] for sample in self.global_dataset.samples])
        indices = np.where(abs_proxy_labels != -1)[0] # remove outliers
        all_proxies = np.unique(abs_proxy_labels[indices])

        # create container
        memory = torch.zeros((all_proxies.size, feature_dim))

        # compute and assign proxy centroids
        for proxy in all_proxies:
            indices = np.where(abs_proxy_labels == proxy)[0]
            proxy_feat = self.features[indices, :].detach().clone()
            proxy_feat = torch.mean(proxy_feat, axis=0) # mean value
            proxy_feat /= torch.norm(proxy_feat) # L2 normalization

            memory[proxy, :] = proxy_feat # save to memory
        
        print('>>> memory bank initiated with {} proxies of {} dims.'.format(all_proxies.size, feature_dim))
        
        
        # import ipdb; ipdb.set_trace()
        # NOTE 2021.3.21
        # Return a cam-to-abs_proxy_label mapper
        mapper = {}
        pseudo_cluster_labels = np.array([sample[3] for sample in self.global_dataset.samples])
        inliner_idx = np.where(pseudo_cluster_labels != -1)[0]
        inliner_cam_abs_proxy = np.array([(int(sample[2].split('_')[-1]), sample[-1]) for sample in self.global_dataset.samples])[inliner_idx,:]
        all_cams = np.unique(inliner_cam_abs_proxy[:,0])
        for cam in all_cams:
            indices = np.where(cam == inliner_cam_abs_proxy[:,0])
            percam_abs_proxy_labels = inliner_cam_abs_proxy[indices,1].squeeze()
            mapper[cam] = sorted(np.unique(percam_abs_proxy_labels).tolist())
        
        print('>>> cam-to-proxy mapper created with {} cams and {} proxies.'.format(len(mapper.keys()), sum([len(value) for value in mapper.values()])))

        return mapper, memory

    def _intra_cam_loss(self, percam_feat, percam_proxies, percam_abs_proxy_label):
        return IntraCamLoss.apply(percam_feat, percam_proxies, percam_abs_proxy_label, self.memory, self.momentum)
    
    def forward(self, batch_feat, abs_proxy_label, camid, pseudo_cluster_label, epoch, global_label, k, inter_loss_epoch):
        # Intra-cam loss
        mapper = self.mapper
        intra_loss = torch.tensor(0.).to(batch_feat.device)
        for c in torch.unique(camid):
            percam_feat_indices = torch.where(camid == c)[0]
            percam_feat = batch_feat[percam_feat_indices,:]
            percam_abs_proxy_label = abs_proxy_label[percam_feat_indices]


            # NOTE 2021.3.21 bug fixed
            all_proxy_labels_percam = torch.tensor(mapper[c.item()]).to(self.memory.device)
            percam_proxies = self.memory[all_proxy_labels_percam,:].clone()


            # get percam targets
            target_mapper = {pid.item():i for i, pid in enumerate(all_proxy_labels_percam)}
            percam_targets = torch.tensor([target_mapper[each.item()] for each in percam_abs_proxy_label]).to(batch_feat.device)

            percam_sim = self._intra_cam_loss(percam_feat, percam_proxies, percam_abs_proxy_label)
            percam_sim /= self.temp
            intra_loss += torch.nn.functional.cross_entropy(percam_sim, percam_targets)

            # Inter-cam loss
            # Inter-cam loss is computed under each camera
            if epoch >= inter_loss_epoch:
                percam_pseudo_cluster_label = pseudo_cluster_label[percam_feat_indices]
                total_inter_loss = torch.tensor(0.).to(batch_feat.device)
                global_cluster_label = torch.tensor([item[3] for item in global_label]).to(batch_feat.device)
                global_camid = torch.tensor([int(item[2].split('_')[-1]) for item in global_label]).to(batch_feat.device)
                global_abs_proxy_label = torch.tensor([item[-1] for item in global_label]).to(batch_feat.device)


                # inter_inputs = torch.mm(percam_feat, self.memory.t().clone())
                # temp_sim = inter_inputs.detach().clone()
                # inter_inputs /= self.temp

                for i, cluster in enumerate(percam_pseudo_cluster_label):
                    # Get positive indices
                    pos_indices = torch.where(cluster == global_cluster_label)[0]
                    diff_cam_indices = torch.where(c != global_camid)[0]
                    pos_indices = _intersection(pos_indices, diff_cam_indices)
                    if pos_indices.size(0) == 0:
                        continue # skip no positive sample cases
                    pos_proxy_label = torch.unique(global_abs_proxy_label[pos_indices])

                    # Get negative indices
                    neg_indices = torch.where(cluster != global_cluster_label)[0]
                    inlier_indices = torch.where(-1 != global_cluster_label)[0]
                    neg_indices = _intersection(neg_indices, inlier_indices) # remove outliers (-1 labeled)
                    neg_proxy_label = torch.unique(global_abs_proxy_label[neg_indices])

                    # Get all positive proxies
                    pos_proxy_feat = self.memory[pos_proxy_label,:]
                    
                    # Get k-nearest negative proxies for hard mining
                    curr_feat = percam_feat[i,:].clone().detach()
                    neg_proxy_feat = self.memory[neg_proxy_label,:]
                    sim = torch.mm(curr_feat.view(1,-1), neg_proxy_feat.t()) # use cosine similarity
                    # dist = torch.norm(curr_feat - neg_proxy_feat, p=2, dim=1)
                    k_nearest_indices = torch.sort(sim.squeeze(), descending=True)[1][k:]
                    neg_proxy_feat = neg_proxy_feat[k_nearest_indices,:]
                    neg_proxy_label = neg_proxy_label[k_nearest_indices] # update negative proxy labels

                    # Compute inter-cam loss
                    inter_proxy_feat = torch.cat([pos_proxy_feat, neg_proxy_feat], dim=0)
                    inter_proxy_label = torch.cat([pos_proxy_label, neg_proxy_label], dim=0)
                    pos_card = pos_proxy_label.size(0)

                    inter_sim = torch.mm(percam_feat[i,:].view(1,-1).clone(), inter_proxy_feat.t()) # (1, num_proxy)
                    inter_sim /= self.temp
                    # inter_sim = torch.exp(inter_sim) # no exponential mapping in official implementation
                    
                    inter_loss = -1.0 * torch.sum(torch.nn.functional.log_softmax(inter_sim, dim=1)[:,:pos_card])
                    inter_loss /= pos_card
                    total_inter_loss += inter_loss

                # import ipdb; ipdb.set_trace()
                # print('inter cam loss: {}'.format((0.5*total_inter_loss/percam_feat.size(0)).item()))
                intra_loss += 0.5 * total_inter_loss / percam_feat.size(0)
        
        return intra_loss
