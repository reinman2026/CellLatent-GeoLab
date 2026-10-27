// Verbatim excerpts from the locally verified source snapshot.
export const verifiedSource = {
  "date": "2026-10-26",
  "sources": {
    "module": {
      "path": "regvelo/src/regvelo/_module.py",
      "sha256": "c5b9ab38c752ebef12df8e594f5327d97507961cbc562220791e3d097d6903b3"
    },
    "model": {
      "path": "regvelo/src/regvelo/_model.py",
      "sha256": "d08df1fe104778ceeb616f2de74404dea25466f441a5e91a7a62131ea60ccc3c"
    },
    "scvi": {
      "path": "scvi-tools-1.2.0/src/scvi/nn/_base_components.py",
      "sha256": "fdbea7d324ec4051471ccffa9bc87fbf365f7832bc6bdc4ae041d411c5f5038d"
    }
  },
  "excerpts": {
    "defaults": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 444,
      "end": 457,
      "text": "        n_hidden: int = 256,\n        n_latent: int = 10,\n        n_layers: int = 1,\n        lam: float = 1,\n        lam2: float = 0,\n        vector_constraint: bool = True,\n        alpha_constraint: float = 0.1,\n        bias_constraint: bool = True,\n        dropout_rate: float = 0.1,\n        log_variational: bool = False,\n        latent_distribution: str = \"normal\",\n        use_batch_norm: Literal[\"encoder\", \"decoder\", \"none\", \"both\"] = \"both\",\n        use_layer_norm: Literal[\"encoder\", \"decoder\", \"none\", \"both\"] = \"both\",\n        var_activation: Callable = torch.nn.Softplus(),"
    },
    "input": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 676,
      "end": 684,
      "text": "        spliced_ = spliced\n        unspliced_ = unspliced\n        if self.log_variational:\n            spliced_ = torch.log(0.01 + spliced)\n            unspliced_ = torch.log(0.01 + unspliced)\n\n        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)\n\n        qz_m, qz_v, z = self.z_encoder(encoder_input)"
    },
    "encoder": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 531,
      "end": 545,
      "text": "        # z encoder goes from the n_input-dimensional data to an n_latent-d\n        # latent space representation\n        n_input_encoder = n_genes\n        self.z_encoder = Encoder(\n            n_input_encoder,\n            n_latent,\n            n_layers=n_layers,\n            n_hidden=n_hidden,\n            dropout_rate=dropout_rate,\n            distribution=latent_distribution,\n            use_batch_norm=use_batch_norm_encoder,\n            use_layer_norm=use_layer_norm_encoder,\n            var_activation=var_activation,\n            activation_fn=torch.nn.ReLU,\n        )"
    },
    "layers": {
      "path": "scvi-tools-1.2.0/src/scvi/nn/_base_components.py",
      "start": 82,
      "end": 96,
      "text": "                            nn.Linear(\n                                n_in + cat_dim * self.inject_into_layer(i),\n                                n_out,\n                                bias=bias,\n                            ),\n                            # non-default params come from defaults in original Tensorflow\n                            # implementation\n                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)\n                            if use_batch_norm\n                            else None,\n                            nn.LayerNorm(n_out, elementwise_affine=False)\n                            if use_layer_norm\n                            else None,\n                            activation_fn() if use_activation else None,\n                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,"
    },
    "heads": {
      "path": "scvi-tools-1.2.0/src/scvi/nn/_base_components.py",
      "start": 283,
      "end": 290,
      "text": "        q = self.encoder(x, *cat_list)\n        q_m = self.mean_encoder(q)\n        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps\n        dist = Normal(q_m, q_v.sqrt())\n        latent = self.z_transformation(dist.rsample())\n        if self.return_dist:\n            return dist, latent\n        return q_m, q_v, latent"
    },
    "time": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 952,
      "end": 968,
      "text": "        \"\"\"\n\n        # predict the abundance in induction phase for target genes\n        ind_t = self.t_max * px_rho\n        n_cells = px_rho.shape[0]\n        mean_u, mean_s, index = self._get_induction_unspliced_spliced(\n            ind_t\n        )\n        \n        ### only consider induction phase\n        scale_u = scale[: self.n_targets, 0].expand(n_cells, self.n_targets).sqrt()\n        scale_s = scale[self.n_targets :, 0].expand(n_cells, self.n_targets).sqrt()\n\n        dist_u = Normal(mean_u, scale_u)\n        dist_s = Normal(mean_s, scale_s)\n        \n        return dist_s, dist_u, index, ind_t"
    },
    "ode": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 291,
      "end": 315,
      "text": "        ## split x into unspliced and spliced readout\n        ## x is a matrix with (G*2), in which row is a subgraph (batch)\n        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)\n        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)\n        if self.activate != \"OR\":\n            if self.base_alpha is not True:\n                grn = self.fc1.weight\n                alpha_unconstr = torch.matmul(s,grn.T)\n            else:\n                alpha_unconstr = self.fc1(s)\n\n            if self.activate == \"softplus\":\n                alpha = torch.clamp(F.softplus(alpha_unconstr),0,50)\n            elif self.activate == \"sigmoid\":\n                alpha = torch.sigmoid(alpha_unconstr)*self.alpha_unconstr_max\n        elif self.activate == \"OR\":\n            alpha = torch.exp(self.fc1(torch.log(torch.nn.functional.softsign(s - 0.5)+1)))\n        else:\n            raise NotImplementedError\n\n        ## Predict velocity\n        du = alpha - beta*u\n        ds = beta*u - gamma*s\n\n        return du,ds"
    },
    "loss": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 864,
      "end": 915,
      "text": "        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)\n\n        reconst_loss_s = -dist_s.log_prob(target_spliced)\n        reconst_loss_u = -dist_u.log_prob(target_unspliced)\n        \n        reconst_loss_target = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)\n        \n        \n        alpha = self.v_encoder.transcription_rate(regulator_spliced)\n        target_unspliced_sort = torch.gather(target_unspliced,0,t_sort) ## measure the lag target unspliced readout (t+1)\n        alpha_sort = torch.gather(alpha,0,t_sort) ## measure the transcription activity (t)\n        alpha_loss = self.alpha_constraint * self.pearson_correlation_loss(target_unspliced_sort,alpha_sort).sum() / alpha.shape[1]\n\n        ## add velocity constraint\n        ## regularize the inferred velocity has both negative and positive compartments\n\n        if self.vector_constraint:\n            du,_ = self.v_encoder(t = 0, u = unspliced, s = spliced)\n            alpha = self.v_encoder.transcription_rate(s = spliced)\n            du = alpha - beta * unspliced\n            velo_loss =  100 * torch.norm(du,dim=1)\n        else:\n            velo_loss = 0\n\n        ## add graph constraint\n        if self.soft_constraint:\n            ## Using norm function to perform graph regularization\n            mask_m = 1 - self.v_encoder.mask_m_raw\n            graph_constr_loss =  self.lamba * torch.norm(self.v_encoder.fc1.weight * mask_m)\n        else:\n            graph_constr_loss = 0\n        \n        Jaco = self.v_encoder.GRN_Jacobian(dist_s.mean.mean(0))\n        loss = torch.nn.L1Loss(reduction = \"sum\")\n        L1_loss = (self.lamba2)*loss(Jaco,self.target_m)\n    \n        ## regularize bias need to be negative\n        if self.bias_constraint:\n            bias_regularize = torch.norm(self.v_encoder.fc1.bias + 10)\n        else:\n            bias_regularize = 0\n        \n        # local loss\n        kl_local = kl_divergence_z\n        weighted_kl_local = (kl_divergence_z) * kl_weight\n        local_loss = torch.mean(reconst_loss_target + weighted_kl_local + velo_loss)\n\n        # total loss\n        loss = local_loss + alpha_loss + L1_loss + graph_constr_loss + bias_regularize\n        \n        loss_recoder = LossOutput(\n            loss=loss, reconstruction_loss=reconst_loss_target, kl_local=kl_local"
    },
    "training": {
      "path": "regvelo/src/regvelo/_model.py",
      "start": 259,
      "end": 284,
      "text": "        self.module.max_epochs = max_epochs\n        if batch_size is None:\n            batch_size = self.batch_size\n\n        user_plan_kwargs = plan_kwargs.copy() if isinstance(plan_kwargs, dict) else {}\n        plan_kwargs = {\"lr\": lr, \"weight_decay\": weight_decay, \"optimizer\": optimizer, \"eps\": eps}\n        plan_kwargs.update(user_plan_kwargs)\n\n        user_train_kwargs = trainer_kwargs.copy()\n        trainer_kwargs = {\"gradient_clip_val\": gradient_clip_val}\n        trainer_kwargs.update(user_train_kwargs)\n\n        data_splitter = DataSplitter(\n            self.adata_manager,\n            train_size=train_size,\n            validation_size=validation_size,\n            batch_size=batch_size,\n        )\n        training_plan = ModifiedTrainingPlan(self.module, **plan_kwargs)\n\n        es = \"early_stopping\"\n        trainer_kwargs[es] = (\n            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]\n        )\n        trainer_kwargs[\"early_stopping_patience\"] = 45\n        trainer_kwargs[\"early_stopping_monitor\"] = 'elbo_validation'"
    },
    "registration": {
      "path": "regvelo/src/regvelo/_model.py",
      "start": 778,
      "end": 808,
      "text": "    def setup_anndata(\n        cls,\n        adata: AnnData,\n        spliced_layer: str = None,\n        unspliced_layer: str = None,\n        **kwargs,\n        ) -> None:\n        r\"\"\"Sets up the AnnData object for use with REGVELOVI.\n\n        This method registers the necessary layers in the AnnData object for use in training\n        and inference.\n\n        Parameters\n        ----------\n        adata\n            Annotated data object with spliced and unspliced layers.\n        spliced_layer\n            Name of the layer in AnnData object that contains spliced normalized expression.\n        unspliced_layer\n            Name of the layer in AnnData object that contains unspliced normalized expression.\n        **kwargs\n            Additional keyword arguments passed to the AnnDataManager.\n        \"\"\"\n        setup_method_args = cls._get_setup_method_args(**locals())\n        anndata_fields = [\n            LayerField(REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),\n            LayerField(REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),\n        ]\n\n        adata_manager = AnnDataManager(\n            fields=anndata_fields, setup_method_args=setup_method_args"
    },
    "global_rates": {
      "path": "regvelo/src/regvelo/_module.py",
      "start": 703,
      "end": 722,
      "text": "            \"alpha_1\": alpha_1,\n        }\n        return outputs\n\n    def _get_rates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n        r\"\"\"Retrieves kinetic parameters from the velocity encoder module.\n\n        Returns\n        -------\n        (``gamma``, ``beta``, ``alpha_1``) parameter vectors for target genes.\n        \"\"\"\n        # globals\n        # degradation for each target gene\n        gamma = torch.clamp(F.softplus(self.v_encoder.gamma_mean_unconstr), 0, 50)\n        # splicing for each target gene\n        beta = torch.clamp(F.softplus(self.v_encoder.beta_mean_unconstr), 0, 50)\n        # transcription for each target gene (bias term)\n        alpha_1 = self.alpha_1_unconstr\n\n        return gamma, beta, alpha_1"
    },
    "posterior_api": {
      "path": "regvelo/src/regvelo/_model.py",
      "start": 532,
      "end": 541,
      "text": "    def rgv_expression_fit(\n        self,\n        adata: AnnData = None,\n        indices: Sequence[int] = None,\n        gene_list: Sequence[str] = None,\n        n_samples: int = 1,\n        batch_size: int = None,\n        return_mean: bool = True,\n        return_numpy: bool = None,\n        ) -> tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]:"
    },
    "posterior_fit": {
      "path": "regvelo/src/regvelo/_model.py",
      "start": 598,
      "end": 626,
      "text": "            for _ in range(n_samples):\n                _, generative_outputs = self.module.forward(\n                    tensors=tensors,\n                    compute_loss=False,\n                )\n                px_rho = generative_outputs[\"px_rho\"]\n\n                ind_t = self.module.t_max * px_rho\n                fit_u, fit_s, _ = self.module._get_induction_unspliced_spliced(\n                    ind_t\n                )\n\n                fit_s = fit_s[..., gene_mask]\n                fit_s = fit_s.detach().cpu().numpy()\n                fit_u = fit_u[..., gene_mask]\n                fit_u = fit_u.detach().cpu().numpy()\n\n                minibatch_samples_s.append(fit_s)\n                minibatch_samples_u.append(fit_u)\n\n            # samples by cells by genes\n            fits_s.append(np.stack(minibatch_samples_s, axis=0))\n            if return_mean:\n                # mean over samples axis\n                fits_s[-1] = np.mean(fits_s[-1], axis=0)\n            # samples by cells by genes\n            fits_u.append(np.stack(minibatch_samples_u, axis=0))\n            if return_mean:\n                # mean over samples axis"
    }
  }
};
