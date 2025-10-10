[![I-CON](https://img.shields.io/badge/I-CON-blue  )](https://mhamilton.net/icon  )
[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue  )](https://t.me/TheWeeklyBrief  )

# I‑CON — A Unifying Framework for Representation Learning

> **One formula → the entire zoo of losses**  
> I‑CON reveals that SNE, t‑SNE, InfoNCE, SupCon, CLIP, k‑Means, CE, and ~20 other methods are all special cases of minimizing the same KL divergence between a "perfect" neighborhood distribution *p* and the "actual" distribution *q*.

## 🚀 Key Achievements

* 🧩 **"Periodic Table" of Methods** — A visual map where switching *p* or *q* lets you "transition" from t‑SNE to SimCLR or from CLIP to SupCon.  
* 📈 **+8 pp over SOTA** in unsupervised classification on ImageNet‑1K via the new Debiased InfoNCE Clustering.  
* 🧹 **α‑Debiasing**: Adding uniform noise to *p* dramatically improves the robustness of contrastive models.  
* 🔄 **Cross-Axis Idea Transfer**: Techniques like label-smoothing from classification work in contrastive learning, and graph-based tricks from DR apply to clustering.

## Why I‑CON Matters?

| Problem                                       | I‑CON Solution                                  |
| --------------------------------------------- | ----------------------------------------------- |
| Disconnected losses for DR / CL / Clustering / CE | Single KL formulation                           |
| Hard to design new methods                    | Simply "combine" distributions                  |
| Overconfidence in contrastive models          | α‑debiasing smooths *p*                         |
| No intrinsic metrics within the loss          | KL divergence provides a natural quality scale  |

## Core Ideas

1. \**Choose the "ideal" $p(j|i)$:  
   Gaussian, k‑NN, one‑hot, cross‑modal pairs — defines "what constitutes neighborhood".  
2. \**Define the "learned" $q(j|i)$:  
   Gaussian / t-distribution in embeddings, uniform over clusters, etc.  
3. **Minimize  $D_{KL}\!\bigl(p(\cdot\!\mid i)\,\Vert\,q(\cdot\!\mid i)\bigr)$.**  
   Obtain SNE, InfoNCE, CE… or a novel hybrid.  
4. **Add α‑debiasing:**  
   $\tilde p=(1-\alpha)p+\alpha/N$ — analog of label-smoothing for any *p*.

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>