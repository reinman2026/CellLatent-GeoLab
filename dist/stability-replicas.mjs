import {paperToyStability} from './paper-toy-stability-results.mjs';

// The default stability page follows the final paper toy: one shared
// six-gene (12-dimensional S/U) observation set, three latent dimensions,
// and five geometry representatives selected from the 20 fitted models.
// Keep the historical wrapper format so older import/validation paths remain
// compatible with this page.
export const stabilityReplicas={
  format:'regvelo-replicas-v1',
  status:paperToyStability.status,
  generatedAt:paperToyStability.generatedAt,
  data:paperToyStability.data,
  expectedModelCount:5,
  models:paperToyStability.representatives.map((model,index)=>({...model,index})),
  plannedSeeds:paperToyStability.representatives.map(model=>model.stabilitySeed??model.seed),
  provenance:{...paperToyStability.provenance,source:'paperToyStability representatives; shared six-gene paper toy observations'}
};
