# Comparative Analysis of Biological and Artificial Neural Networks

## Introduction
Neural networks are fundamental structures underlying cognitive processes in biological organisms and have inspired scientists to create mathematical models known as artificial neural networks. Comparing these two approaches provides insight into the nature of thinking, learning systems, and the possibility of replicating intelligent behavior. This text provides a detailed examination of the basic principles of biological neural network organization and artificial system architectures, highlighting their similarities and differences in terms of structure, dynamics, learning, and computational capabilities.

## **Biological Neuron and Its Composition**

### Structure of the Biological Neuron

Biological neural networks represent the fundamental basis of the nervous system in living organisms, ensuring the processing, transmission, and storage of information. Their structural and functional unit is the neuron—a highly specialized cell designed to generate and propagate electrochemical signals. The term "neuron" for nerve cells was introduced by G. V. Waldeyer in 1891 [0]. Understanding the structure and function of neurons is critical for comprehending the principles of biological neural networks and their modeling in artificial systems, such as artificial neural networks (ANNs), which were initially inspired by biological analogs.

A neuron consists of several key elements:

1. **Soma (cell body):** Contains the nucleus and major organelles, providing metabolic and regulatory processes, and is responsible for the nutrition, resource management, and other vital functions of the cell.
2. **Dendrites:** Short, branching structures through which the neuron receives input signals from other cells. Synaptic contacts are primarily formed on dendrites. The neuron possesses the important property of summation, i.e., the ability to receive multiple excitations through dendrites and integrate them into a single signal.
3. **Axon:** The single, long projection through which the action potential (spike) is transmitted over long distances to synapses. At the end of the axon is a synapse—the site of contact between neurons.
4. **Synapses:** The site of chemical or electrical signal transmission. Synaptic vesicles release neurotransmitters (glutamate, GABA, acetylcholine, etc.), which bind to receptors on the postsynaptic membrane. A distinctive feature of the synapse is its unidirectionality: the signal is transmitted only from the axon to the dendrite of the next neuron.

A neuron can be in two states—excitation and inhibition. The ability of neurons to sum and transmit signals through complex networks of interactions underlies the functioning of the nervous system and explains the complexity of interactions within the organism, as well as our mental capabilities. These principles, borrowed from biology, laid the foundation for the development of artificial neural networks, which model information processing processes similar to those occurring in biological systems.

![Structure of the biological neuron](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-01-02/assets/Page_1.png)

**General Mathematical Model of the Neuron:**

The fundamental and widely used mathematical model describing the dynamics of the action potential in living neurons, including human ones, is the **Hodgkin-Huxley model**. This model, developed based on experiments with the giant axon of the squid, describes the change in the membrane potential of the neuron over time, taking into account the flow of sodium ($Na^+$) and potassium ($K^+$) ions through voltage-gated ion channels, as well as the leakage current. [1](https://www.nature.com/articles/nn1100_1165)

The main equation of the model represents a balance of currents through the membrane:

$$C_m \frac{dV_m}{dt} = -I_{Na} - I_K - I_L + I_{ext}$$

where:
- $C_m$ — membrane capacitance (F/m²).
- $\frac{dV_m}{dt}$ — the rate of change of the membrane potential $V_m$ (V).
- $I_{Na}$, $I_K$ — the currents of sodium and potassium ions, respectively (A/m²).
- $I_L$ — the leakage current (A/m²).
- $I_{ext}$ — the external applied current (A/m²).

The ion currents are described by the following equations:

$$I_{Na} = g_{Na} m^3 h (V_m - E_{Na})$$

$$I_K = g_K n^4 (V_m - E_K)$$

$$I_L = g_L (V_m - E_L)$$

where:
- $g_{Na}$, $g_K$, $g_L$ — the maximum conductances for sodium, potassium, and leakage currents, respectively (S/m²). These parameters reflect the density of the corresponding ion channels on the membrane.
- $E_{Na}$, $E_K$, $E_L$ — the equilibrium potentials (Nernst potentials) for sodium, potassium, and leakage currents, respectively (V). They are determined by the concentration of ions inside and outside the cell.
- $m$ — the activation variable of sodium channels (dimensionless, from 0 to 1). Describes the probability of sodium channel opening.
- $h$ — the inactivation variable of sodium channels (dimensionless, from 0 to 1). Describes the probability that sodium channels will not be blocked.
- $n$ — the activation variable of potassium channels (dimensionless, from 0 to 1). Describes the probability of potassium channel opening.

The activation and inactivation variables $m$, $h$, and $n$ are functions of membrane potential and time and are described by first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_m) (1-m) - \beta_m(V_m) m$$

$$\frac{dh}{dt} = \alpha_h(V_m) (1-h) - \beta_h(V_m) h$$

$$\frac{dn}{dt} = \alpha_n(V_m) (1-n) - \beta_n(V_m) n$$

where $\alpha(V_m)$ and $\beta(V_m)$ — the voltage-dependent opening and closing rates of ion channels, which are determined experimentally.

**Biological Description of the Hodgkin-Huxley Model:**

In the resting state, the neuron maintains certain ionic gradients (high potassium concentration inside and sodium concentration outside), creating a negative membrane potential (approximately -70 mV) inside the cell relative to the external environment. When the membrane potential becomes less negative and reaches a certain threshold, usually around -55 mV, this causes a rapid opening of voltage-gated sodium channels, and positive sodium ions rush into the soma (more precisely, into the axon). This movement of ions is caused by the electrochemical gradient, including the difference in concentrations and electrical charge. This causes a rapid depolarization, and the electrical charge inside the axon of the neuron becomes positive. Soon after this, sodium channels inactivate, preventing further sodium influx. At this point, slower potassium channels open, and positive potassium ions begin to exit outward. This outflow of positive charge leads to repolarization, returning the membrane potential to negative values. Sometimes, a brief hyperpolarization may occur. This happens because some voltage-gated potassium channels remain open longer than necessary to return to the resting potential, temporarily making the membrane potential more negative than usual. Eventually, the sodium-potassium pump (actively pumping sodium out of the cell and potassium into the cell) helps restore and maintain the ionic gradients, returning the axon of the neuron to a stable negative resting potential. The described processes occur mainly in the axon hillock and axon of the neuron, where the action potential is generated and propagated.

**Signal Transmission within Neurons:**

Transmission of signals within a neuron (over long distances):

*   The action potential is an active way of signal transmission. As soon as the action potential is triggered at the beginning of the axon (axon hillock), it self-reproduces along the entire length of the axon to the very nerve endings. This occurs due to the sequential opening and closing of voltage-gated ion channels along the axon. Each "point" of the axon generates a new action potential, as if "pushing" the signal further.

Transmission of signals between neurons (synaptic transmission):

*   When the action potential reaches the nerve endings (presynaptic terminals) of the axon, it triggers the process of synaptic transmission.
*   Depolarization of the presynaptic terminal (caused by the arriving action potential) leads to the opening of voltage-gated calcium channels.
*   Calcium ions ($Ca^{2+}$) enter the presynaptic terminal.
*   The influx of calcium causes synaptic vesicles containing neurotransmitters to fuse with the presynaptic membrane.
*   Neurotransmitters are released into the synaptic cleft (the space between two neurons).
*   Neurotransmitters bind to receptors on the membrane of the next neuron (postsynaptic neuron).
*   Binding of neurotransmitter to receptor can cause a change in the ion permeability of the postsynaptic membrane, which in turn can lead to depolarization (excitatory postsynaptic potential - EPSP) or hyperpolarization (inhibitory postsynaptic potential - IPSP) of this neuron.
*   If the sum of depolarization of the postsynaptic neuron reaches the threshold, an action potential will also be generated in it, and the signal will be transmitted further.

Thus, the action potential performs two key functions in signal transmission:

*   Ensures reliable transmission of the signal over long distances within one neuron. Without this, the signal would simply attenuate.
*   Triggers the process of synaptic transmission, allowing neurons to "communicate" with each other. It is precisely the action potential that acts as the "trigger" for the release of neurotransmitters.

**Relevance to Human Neurons:**

Although the original model was developed for the axon of the squid, its principles underlie the understanding of action potential generation in most excitable cells, including human neurons. The parameters of the model ($g_{Na}$, $g_K$, $g_L$, the kinetics of variables $m$, $h$, $n$) may vary depending on the type of neuron and its functional characteristics. Modern research actively works on adapting and parameterizing the Hodgkin-Huxley model for various types of human neurons, using experimental data obtained using patch-clamp methods.

**Significance and Limitations:**

The Hodgkin-Huxley model is fundamental because it:

*   **Is based on biophysical processes:** Describes the behavior of ion channels, which are key elements in generating the action potential.
*   **Quantitatively reproduces the shape of the action potential:** Can accurately model the dynamics of the membrane potential.
*   **Serves as the basis for more complex models:** Many modern neuron models are built on the principles laid out in the Hodgkin-Huxley model.

However, it is important to note its limitations:

*   **Describes a point neuron:** Does not account for the spatial structure of the neuron (dendrites, axon) and the propagation of signals along them. For modeling these aspects, multi-compartment models are used.
*   **Simplifies ion channel behavior:** Represents ion channels as homogeneous populations, not accounting for their subunit structure and stochastic opening/closing.
*   **Does not include all types of ion channels:** Real neurons express a variety of ion channels that can influence their excitability and firing patterns.
*   **Parameters can be difficult to determine:** Accurately determining the parameters of the model for a specific human neuron requires complex experimental studies.

**Modern Extensions:**

Currently, there are many extensions of the Hodgkin-Huxley model that aim to overcome its limitations and more accurately model neuron behavior:

*   **Multi-compartment models:** Divide the neuron into several compartments (soma, dendrites, axon) and model the propagation of electrical signals between them.
*   **Models including various types of ion channels:** Add equations to describe the dynamics of other types of ion channels (e.g., calcium, chloride, various types of potassium channels).
*   **Models incorporating synaptic transmission:** Integrate mechanisms of synaptic transmission, allowing modeling of interactions between neurons.
*   **Models accounting for intracellular processes:** Include biochemical processes affecting the function and excitability of the neuron.

In conclusion, the Hodgkin-Huxley model, despite its age, remains a cornerstone in the mathematical modeling of neurons, including human ones. Understanding its principles and limitations is essential for interpreting modeling results and developing more complex and realistic models.

## Soma (Cell Body): Center of Metabolism and Signal Integration

The soma, or cell body, is the central part of the neuron, containing the nucleus and major organelles necessary for the cell's life support.

**Internal Structure of the Neuron:**
*   **Nucleus:** Contains the neuron's genetic material (DNA), organized into chromosomes. The nucleus controls the synthesis of proteins necessary for the structure and function of the neuron, including enzymes, structural proteins, and receptors. Transcription of DNA into mRNA and subsequent translation on ribosomes ensure the continuous renewal and maintenance of cellular components. Regulation of gene expression in the nucleus plays a key role in the neuron's adaptation to changing conditions and in processes of learning and memory.
*   **Organelles:** The soma contains the typical set of eukaryotic organelles, each performing specialized functions:
    *   **Mitochondria:** Responsible for producing energy in the form of ATP through cellular respiration. Neurons have high metabolic activity and require significant amounts of energy to maintain ion gradients and signal transmission processes.
    *   **Endoplasmic Reticulum (ER):** Involved in the synthesis and transport of proteins and lipids. The rough ER, covered with ribosomes, synthesizes proteins destined for membranes and secretion. The smooth ER is involved in lipid synthesis and detoxification.
    *   **Golgi Apparatus:** Modifies, sorts, and packages proteins and lipids for transport to other parts of the cell or for secretion. Participates in the formation of lysosomes and synaptic vesicles.
    *   **Lysosomes:** Contain hydrolytic enzymes necessary for breaking down cellular waste and damaged organelles (autophagy).
    *   **Cytoskeleton:** Composed of a network of protein filaments (microtubules, actin filaments, neurofilaments), providing structural support to the neuron, maintaining its shape, and participating in the transport of substances within the cell (axonal transport).
*   **Cytoplasm:** The gel-like substance filling the soma, in which organelles are located and various molecules, including ions, enzymes, and nutrients, are dissolved.

The soma also plays an important role in integrating incoming signals. The potentials generated on the dendrites propagate to the soma, where their summation occurs. If the sum reaches the threshold potential at the axon hillock, an action potential is generated.

**Soma (Cell Body) - Mathematical Model:**

To accurately and realistically describe the integrative function of the soma, it is necessary to consider that the neuronal membrane is not a passive element. **Active properties of the membrane**, due to the presence of various voltage-gated ion channels, play a key role in processing incoming signals. Within the **single-compartment approach**, which assumes that the potential within the soma is uniformly distributed, the mathematical formalization of the dynamics of the soma membrane potential relies on the following principles:

$$C_m \frac{dV_{soma}}{dt} = -I_{ion} + I_{syn}$$

where:

*   $C_m$ — membrane capacitance of the soma (F).
*   $\frac{dV_{soma}}{dt}$ — the rate of change of the soma membrane potential (V/s).
*   $I_{ion}$ — the total ionic current through the soma membrane (A). This current includes the currents of various ions, such as sodium, potassium, calcium, and others, depending on the type of neuron.
*   $I_{syn}$ — the total synaptic current entering the soma from the dendrites (A).

**Detailing the ionic current ($I_{ion}$):**

The ionic current represents the sum of currents from individual ions passing through various types of ion channels present in the soma membrane. For example, consider the main currents present in most neurons:

$$I_{ion} = I_{Na} + I_K + I_L + I_{Ca} + ...$$

where:

*   $I_{Na}$ — the current of sodium ions through voltage-gated sodium channels.
*   $I_K$ — the current of potassium ions through various types of voltage-gated potassium channels (e.g., delayed rectifier, A-type, etc.).
*   $I_L$ — the leakage current, representing the passive current of ions through the membrane.
*   $I_{Ca}$ — the current of calcium ions through voltage-gated calcium channels (various types, such as L-type, N-type, T-type, P/Q-type).
*   `...` — indicates the possibility of other ionic currents, such as chloride currents, currents activated by hyperpolarization (Ih), and others, depending on the type of neuron.

Each ionic current is described by an equation of the form:

$$I_{ion\_type} = g_{ion\_type} \cdot m^p \cdot h^q \cdot (V_{soma} - E_{ion\_type})$$

where:

*   $g_{ion\_type}$ — the maximum conductance for this type of ion channels (S). This parameter reflects the density of channels on the membrane.
*   $m$ — the activation variable of the channel (dimensionless, from 0 to 1), describing the probability of the channel opening.
*   $h$ — the inactivation variable of the channel (dimensionless, from 0 to 1), describing the probability that the channel will not be blocked. Not all channel types have an inactivation variable.
*   $p$ and $q$ — stoichiometric coefficients, determining the number of activation and inactivation molecules required to open the channel.
*   $V_{soma}$ — the membrane potential of the soma (V).
*   $E_{ion\_type}$ — the equilibrium potential (Nernst potential) for this ion (V).

The activation variables ($m$) and inactivation variables ($h$) are functions of the membrane potential and time and are described by first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_{soma}) (1-m) - \beta_m(V_{soma}) m$$
$$\frac{dh}{dt} = \alpha_h(V_{soma}) (1-h) - \beta_h(V_{soma}) h$$

where $\alpha(V_{soma})$ and $\beta(V_{soma})$ — the voltage-dependent opening and closing rates of ion channels, which are determined experimentally and can have various functional forms.

**Detailing the synaptic current ($I_{syn}$):**

The synaptic current represents the sum of currents from all synapses located on the soma (and, in more general models, on dendrites affecting the soma potential):

$$I_{syn} = \sum_{i} I_{syn\_i}$$

Each synaptic current can be modeled as:

$$I_{syn\_i} = g_{syn\_i}(t) \cdot (V_{soma} - E_{syn\_i})$$

where:

*   $g_{syn\_i}(t)$ — the conductance of the $i$-th synapse, which depends on time and is determined by the release of neurotransmitter and binding to receptors on the postsynaptic membrane. This conductance is usually modeled as a function having a rapid rise and slow decay.
*   $V_{soma}$ — the membrane potential of the soma (V).
*   $E_{syn\_i}$ — the equilibrium potential for this type of synapse (V). For excitatory synapses (e.g., glutamatergic), $E_{syn}$ is usually close to 0 mV, while for inhibitory synapses (e.g., GABAergic), $E_{syn}$ is close to the chloride potential or the resting potential.

The conductance of the synapse $g_{syn\_i}(t)$ is often modeled using kinetic schemes describing the binding of neurotransmitter to receptors and the opening of ion channels, or using alpha functions or their variants.

**Threshold for action potential generation:**

In the single-compartment model, the generation of the action potential is usually not modeled in detail in the soma. Instead, it is assumed that if the soma membrane potential $V_{soma}(t)$ reaches a certain threshold value $\theta_{AP}$, an action potential is initiated in the axon hillock. However, more complex models may include detailed modeling of the ion channels in the axon hillock to accurately reproduce the process of spike initiation.

**Advantages and Limitations of this Model:**

*   **Advantages:**
    *   Takes into account the active properties of the soma membrane, allowing modeling of the generation of local potentials and more realistically describing signal integration.
    *   Can reproduce various firing patterns of the neuron depending on the set of ion channels and their parameters.
*   **Limitations:**
    *   Assumes the soma is isopotential, which is not always true for neurons with large or complex somas.
    *   Simplifies the description of ion channels, representing them as homogeneous populations, not accounting for their subunit structure and stochastic opening/closing.
    *   Does not include all types of ion channels; real neurons express a variety of ion channels that can influence their excitability and firing patterns.
    *   Parameters of the model must be determined experimentally for a specific type of neuron, which can be a difficult task.

**More Complex Models:**

To more accurately model the electrical activity of the soma, especially when the assumption of isopotentiality does not hold, **multi-compartment models** are used. In these models, the soma is divided into several connected compartments, each described by a set of equations similar to those presented above, taking into account the passive properties of the membrane and the connections between compartments.

**Conclusion:**

The presented mathematical model of the soma, based on the single-compartment approach with consideration of active membrane properties, is much more relevant and complete than the simple model of potential summation. It allows modeling the dynamics of the soma membrane potential, determined by the activity of various ion channels and synaptic inputs. For scientific work requiring a detailed understanding of soma electrical activity, using this model or its multi-compartment extensions is necessary. It is important to note that the accuracy of the model directly depends on correct parameterization based on experimental data for a specific type of neuron.

## Dendrites: Reception and Integration of Incoming Signals

Dendrites are numerous, usually short, branching processes extending from the soma. Their main function is to receive signals from other neurons.

**Internal Structure of the Dendrite:**
*   **Dendritic Morphology:** The morphology of dendrites varies significantly depending on the type of neuron and its function. The complex branching structure increases the surface area for receiving synaptic inputs.
*   **Dendritic Spines:** On the surface of many dendrites are small protrusions called dendritic spines. Spines are the primary sites for forming excitatory synapses. The dynamic structure of spines (changes in shape and number) plays an important role in synaptic plasticity, underlying the processes of learning and memory. A spine consists of a head, neck, and base, and its morphology affects the electrical properties of the synaptic signal.
*   **Synaptic Contacts:** Most synaptic contacts are formed on dendrites, both on the smooth surface and on spines. The presynaptic neuron releases neurotransmitters, which bind to receptors on the postsynaptic membrane of the dendrite, causing a change in membrane permeability for ions.
*   **Local Potentials:** Binding of neurotransmitters to receptors on the dendrite causes the generation of local membrane potential changes—postsynaptic potentials (PSP). Excitatory PSPs (EPSPs) depolarize the membrane, bringing it closer to the action potential threshold. Inhibitory PSPs (IPSPs) hyperpolarize the membrane, moving it away from the threshold.
*   **Signal Integration:** Dendrites not only receive signals but also participate in their integration. Spatial summation occurs when signals from several synapses, located on different parts of the dendrite, reach the soma almost simultaneously. Temporal summation occurs when successive signals from the same synapse reach the soma with a short delay. The shape and electrical properties of dendrites (e.g., presence of voltage-dependent ion channels) affect the efficiency of signal integration.

**Dendrites - Mathematical Model:**

**Fundamental Membrane Potential Equation** [2]

The dynamics of the membrane potential in any point of the dendrite can be described based on the law of conservation of charge and modeling the membrane as an electrical circuit:

$$C_m \frac{dV_m(x,t)}{dt} = -I_{ion}(x,t) + I_{ext}(x,t)$$

where:
*   $V_m(x,t)$ - membrane potential at point $x$ at time $t$.
*   $C_m$ - membrane capacitance per unit area.
*   $I_{ion}(x,t)$ - total ionic current through the membrane at point $x$ at time $t$.
*   $I_{ext}(x,t)$ - external current applied to the membrane at point $x$ at time $t$ (e.g., synaptic current).

The ionic current $I_{ion}$ includes the currents of leakage and currents through various ion channels:

$$I_{ion}(x,t) = g_L (V_m(x,t) - E_L) + \sum_i I_i(x,t)$$

where:
*   $g_L$ - leakage conductance.
*   $E_L$ - leakage reversal potential.
*   $I_i(x,t)$ - current through the $i$-th type of ion channels, which may depend on voltage and ion concentration.

**Modeling Synaptic Current**

The synaptic current $I_{syn}(t)$, being a special case of $I_{ext}(x,t)$, arises when neurotransmitter binds to receptors on the postsynaptic membrane. It can be modeled in various ways reflecting the kinetics of binding and channel conductance:

*   **Instantaneous Conductance Jump:**

    $$I_{syn}(t) = g_{syn} \cdot s(t) \cdot (V_m(t) - E_{syn})$$

    where:
    *   $g_{syn}$ - maximum synaptic conductance.
    *   $s(t)$ - binary function equal to 1 during synaptic activity and 0 otherwise.
    *   $E_{syn}$ - synaptic reversal potential.

*   **Model with First-Order Kinetics:**

    $$I_{syn}(t) = g_{syn} \cdot r(t) \cdot (V_m(t) - E_{syn})$$

    where $r(t)$ describes the fraction of open ion channels and follows the differential equation:

    $$\frac{dr}{dt} = \alpha [T](t) (1-r) - \beta r$$

    where:
    *   $\alpha$ - binding rate of neurotransmitter.
    *   $\beta$ - unbinding rate of neurotransmitter.
    *   $[T](t)$ - neurotransmitter concentration in the synaptic cleft.

**Derivation of Postsynaptic Potential (PSP) Models**

The PSP models presented earlier are approximations of membrane potential dynamics in response to synaptic current. They can be derived from more fundamental equations under certain assumptions.

*   **Exponential PSP Model:** This model arises under the assumption that synaptic conductance increases and decreases exponentially. Considering a simplified membrane model with leakage and synaptic current, one can obtain a solution of the form:

    $$PSP(t) \propto e^{-t/\tau_{m}} \int_0^t I_{syn}(\tau) e^{\tau/\tau_{m}} d\tau$$

    where $\tau_m = C_m / g_L$ - membrane time constant. Under approximation of the synaptic current shape, one can obtain expressions close to the previously presented exponential model.

*   **Alpha-function PSP:** This model is often used to describe PSPs mediated by AMPA receptors. It can be obtained from the first-order kinetic model under certain assumptions about the form of $[T](t)$.

**Important to note:** These models are phenomenological and do not always accurately reflect the complex kinetics of ion channels and receptors.

**Spatial Integration of Signals and Cable Theory**

Dendrites are not isopotential structures. Incoming synaptic signals propagate along dendritic branches, attenuating as they move away from the point of origin. Cable theory is used to describe this process.

A dendritic branch is modeled as a cylindrical cable with specific membrane resistance $R_m$, axial resistance $R_a$, and specific membrane capacitance $C_m$. Changes in membrane potential $v(x,t)$ along the dendrite are described by the **cable equation**:

$$\lambda^2 \frac{\partial^2 v(x,t)}{\partial x^2} - \tau_m \frac{\partial v(x,t)}{\partial t} - v(x,t) = r_m i_{ext}(x,t)$$

where:
*   $v(x,t) = V_m(x,t) - V_{rest}$ - deviation of membrane potential from resting potential.
*   $\lambda = \sqrt{\frac{r_m}{r_a}}$ - **spatial constant** (length of attenuation), determining how far the signal propagates along the dendrite. Here $r_m = R_m / \pi d$, $r_a = R_a / (\pi d^2 / 4)$, where $d$ is the diameter of the dendrite.
*   $\tau_m = r_m c_m$ - **membrane time constant**. Here $c_m = C_m \pi d$.
*   $r_m$ - membrane resistance per unit length.
*   $i_{ext}(x,t)$ - external current per unit length (e.g., synaptic current).

Solving this equation allows calculating the contribution of each synaptic input to the potential at any point of the dendrite, including the soma. Spatial summation arises as a linear superposition of these contributions (in the case of passive dendrites).

**Influence of Dendritic Morphology**

The complex branching structure of dendritic trees significantly affects signal integration. Cable theory can be extended to analyze branched structures by considering each branch as a separate cable with appropriate boundary conditions at the branching points.

*   **Input Resistance:** Dendritic morphology determines its input resistance, which affects the amplitude of PSPs. More branched dendrites with smaller branch diameters typically have higher input resistance.
*   **Electrotonic Distance:** The effective distance between synapses and the soma is determined not only by physical distance but also by electrotonic distance, which accounts for signal attenuation. Synapses located on distal branches have less influence on the soma potential due to greater electrotonic distance.

**Temporal Integration of Signals**

Temporal summation occurs when successive PSPs from one or more synapses reach the soma with a short delay. Mathematically, this can be described as the superposition of PSPs, taking into account their temporal dynamics:

$$V_{soma}(t) = \sum_i w_i \cdot PSP_i(t - t_i)$$

where:
*   $w_i$ - effective weight of the $i$-th synapse, taking into account its strength and electrotonic distance to the soma.
*   $PSP_i(t - t_i)$ - shape of the PSP caused by the $i$-th synapse, taking into account the time of its activation $t_i$.

For more accurate modeling, it is necessary to consider the interaction between successive PSPs, especially if they occur rapidly enough to overlap.

**Active Dendrites: Accounting for Nonlinearities**

Unlike passive cables, dendrites of many neuron types contain **voltage-gated ion channels**. These channels introduce nonlinearity into the signal integration process, allowing dendrites to generate local spikes or amplify incoming signals.

Modeling active dendrites requires incorporating Hodgkin-Huxley equations or their modifications to describe the dynamics of various ion channels (e.g., sodium, potassium, calcium):

$$C_m \frac{dV_m}{dt} = -g_L(V_m - E_L) - g_{Na} m^3 h (V_m - E_{Na}) - g_K n^4 (V_m - E_K) - ... + I_{syn}$$

where $m$, $h$, $n$ - activation and inactivation variables of ion channels, described by first-order differential equations.

The presence of active channels allows dendrites to perform more complex computational operations, such as threshold logic at the level of individual branches.

**Compartmental Modeling**

For modeling complex dendritic trees with nonlinear properties, **compartmental modeling** is often used. The dendritic tree is divided into a number of small isopotential segments (compartments). Each compartment is described by a system of differential equations, taking into account currents through the membrane and currents between neighboring compartments.

This approach allows numerically solving the equations for complex geometries and including various types of ion channels in different parts of the dendrite.

**Influence of Dendritic Spines**

Dendritic spines, being the primary sites of most excitatory synapses, play an important role in synaptic transmission and plasticity. Their morphology (size of the head and neck) affects:

*   **Input Resistance:** A spine with a narrow neck increases local input resistance, leading to greater depolarization upon synaptic current arrival.
*   **Electrical Isolation:** The neck of the spine can electrically isolate the spine head from the main dendritic branch, allowing spines to function as relatively independent computational units.
*   **Calcium Diffusion:** Spine morphology affects the diffusion of calcium ions, which play a key role in synaptic plasticity.

Mathematical modeling of spines often involves considering them as additional compartments with specific geometric and electrical properties.

**Stochasticity of Synaptic Transmission**

It is important to note that synaptic transmission is a stochastic process. The release of neurotransmitter and the opening of ion channels occur with a certain probability. For more realistic modeling, probabilistic models of synaptic transmission can be used, for example, Poisson models for neurotransmitter release.

# Axon: Transmission of the Action Potential over Long Distances

The axon is typically a single, long, and thin process extending from the soma at the axon hillock. Its primary function is to transmit the electrical signal (action potential or spike) over long distances to other neurons, muscle cells, or glands.

**Internal Structure of the Axon:**
*   **Axon Hillock:** A specialized region of the soma where incoming signals are integrated and the action potential is initiated. The axon hillock contains a high concentration of voltage-gated sodium channels, making it the most excitable part of the neuron.
*   **Initial Segment of the Axon:** Directly adjacent to the axon hillock and is the site where the action potential is typically generated.
*   **Myelin Sheath:** In many neurons, the axon is covered by a myelin sheath formed by glial cells (oligodendrocytes in the central nervous system and Schwann cells in the peripheral nervous system). Myelin is a lipid insulation that significantly increases the speed of action potential conduction.
*   **Nodes of Ranvier:** Periodic breaks in the myelin sheath where the axon remains uncovered. Voltage-gated sodium channels are concentrated at the nodes of Ranvier, allowing the action potential to "jump" from one node to the next (saltatory conduction), significantly increasing the speed of signal transmission.
*   **Axon Terminals (Presynaptic Terminals):** At the distal end of the axon, it branches into numerous terminals. Each terminal forms a synapse with another cell.
*   **Synaptic Vesicles:** Inside the axon terminals are synaptic vesicles – membrane-bound structures containing neurotransmitters.
*   **Mechanism of Action Potential Transmission:** When the action potential reaches the axon terminal, it causes depolarization of the membrane, leading to the opening of voltage-gated calcium channels. The influx of calcium ions into the terminal triggers the process of synaptic vesicle fusion with the presynaptic membrane (exocytosis) and the release of neurotransmitters into the synaptic cleft.

**Axon - Mathematical Model:**

**Generation of the Action Potential: Hodgkin-Huxley Model**

The fundamental model describing the generation of the action potential is the **Hodgkin-Huxley (HH) model**. It is based on experimental data on ion currents in the giant axon of the squid and describes the dynamics of the membrane potential $V_m$ using a system of nonlinear differential equations:

$$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$

where $I_{ion}$ is the total ionic current, including sodium ($I_{Na}$), potassium ($I_K$), and leakage ($I_L$) currents:

$$I_{ion} = I_{Na} + I_K + I_L$$

Each ionic current is modeled as:

$$I_{Na} = g_{Na} m^3 h (V_m - E_{Na})$$
$$I_K = g_K n^4 (V_m - E_K)$$
$$I_L = g_L (V_m - E_L)$$

where:
*   $g_{Na}$, $g_K$, $g_L$ - maximum conductances for sodium, potassium, and leakage channels, respectively.
*   $E_{Na}$, $E_K$, $E_L$ - equilibrium potentials for sodium, potassium, and leakage, respectively.
*   $m$, $h$, $n$ - dimensionless variables describing the activation of sodium channels, inactivation of sodium channels, and activation of potassium channels, respectively. They follow first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_m) (1-m) - \beta_m(V_m) m$$
$$\frac{dh}{dt} = \alpha_h(V_m) (1-h) - \beta_h(V_m) h$$
$$\frac{dn}{dt} = \alpha_n(V_m) (1-n) - \beta_n(V_m) n$$

where $\alpha(V_m)$ and $\beta(V_m)$ - voltage-dependent opening and closing rates of the channels, determined experimentally.

**Simplifications and Alternative Models for Action Potential Generation**

The Hodgkin-Huxley model, despite its accuracy, is computationally expensive. Simpler models are used for certain tasks:

*   **FitzHugh-Nagumo Model:** A two-dimensional model describing the dynamics of the membrane potential and a recovery variable, mimicking the inactivation of sodium channels and activation of potassium channels.

    $$\frac{dV}{dt} = c(V - \frac{V^3}{3} + R + I_{ext})$$
    $$\frac{dR}{dt} = - \frac{1}{c} (V - a + bR)$$

    where $a$, $b$, $c$ - parameters of the model.

*   **Integrate-and-Fire Models (Integrate-and-Fire):** Abstract models where the membrane potential integrates the incoming current, and when it reaches a threshold, a spike is generated, after which the potential resets. Various versions exist, including adaptation or subthreshold oscillations.

    $$ \tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R I_{ext}$$
    If $V_m(t) \geq V_{thresh}$, then $V_m \rightarrow V_{reset}$.

The choice of model depends on the required accuracy and available computational resources.

**Propagation of the Action Potential in a Non-myelinated Axon**

The propagation of the action potential along the axon can be described using **cable theory**, similar to dendrites, but with the presence of voltage-gated ion channels. The equation for the membrane potential $v(x,t)$ along the axon takes the form:

$$c_m \frac{\partial v}{\partial t} = \frac{1}{r_a} \frac{\partial^2 v}{\partial x^2} - I_{ion}(v)$$

where $I_{ion}(v)$ is the nonlinear ionic current described by the Hodgkin-Huxley equations (or another ion channel model). This is a nonlinear parabolic partial differential equation, and its analytical solution is difficult. Numerical methods are usually used to solve it.

**Saltatory Conduction in Myelinated Axons**

The myelin sheath significantly increases the speed of action potential conduction through **saltatory conduction**. Myelin acts as an insulator, increasing membrane resistance and decreasing membrane capacitance in the internodes. The action potential is generated only at the **Nodes of Ranvier**, where voltage-gated sodium channels are concentrated.

**Mathematical Model of Saltatory Conduction**

Modeling saltatory conduction involves considering the properties of both myelinated internodes and unmyelinated nodes of Ranvier.

*   **Internodes:** In internodes, the dynamics of the membrane potential are described by the cable equation without active ionic currents, but with high $r_m$ and low $c_m$ values.
*   **Nodes of Ranvier:** In the nodes, the dynamics of the membrane potential are described by the Hodgkin-Huxley model (or its analog), as this is where the action potential is generated.

The myelinated axon model can be represented as a series of connected segments representing internodes and nodes of Ranvier. The current propagates passively through the internodes and is actively amplified at the nodes.

Mathematically, this can be described by a system of coupled differential equations for each segment. For the $i$-th node of Ranvier:

$$C_{m,node} \frac{dV_{m,i}}{dt} = -I_{ion,node}(V_{m,i}) + I_{axial,i-1 \rightarrow i} - I_{axial,i \rightarrow i+1}$$

where $I_{axial}$ is the axial current between segments, which depends on the potential difference and the axial resistance between them.

For the $j$-th internode:

$$C_{m,internode} \frac{dV_{m,j}}{dt} = \frac{V_{m,j-1} - V_{m,j}}{R_{axial,j-1 \rightarrow j}} - \frac{V_{m,j} - V_{m,j+1}}{R_{axial,j \rightarrow j+1}}$$

where $R_{axial}$ is the axial resistance of the internode.

**Transmission of Signal at the Axon Terminal: Synaptic Transmission**

When the action potential reaches the axon terminal, it triggers the process of **synaptic transmission**. Depolarization of the presynaptic membrane leads to the opening of voltage-gated calcium channels.

The dynamics of intracellular calcium concentration $[Ca^{2+}]_i$ in the terminal can be described by the equation:

$$\frac{d[Ca^{2+}]_i}{dt} = - \beta I_{Ca} - \frac{[Ca^{2+}]_i - [Ca^{2+}]_{rest}}{\tau_{Ca}}$$

where:
*   $\beta$ - a constant linking the calcium current to the change in concentration.
*   $I_{Ca}$ - the current through calcium channels, modeled similarly to sodium and potassium channels in the HH model.
*   $[Ca^{2+}]_{rest}$ - the resting calcium concentration.
*   $\tau_{Ca}$ - the time constant for calcium buffering and removal.

The influx of calcium ions triggers the fusion of synaptic vesicles with the presynaptic membrane and the release of neurotransmitter into the synaptic cleft.

**Modeling Neurotransmitter Release**

Neurotransmitter release is a probabilistic process. The probability of release $P_{release}$ depends on the concentration of calcium in the presynaptic terminal. Various models can be used:

*   **Threshold Model:** Release occurs when the calcium concentration exceeds a certain threshold.
*   **Sigmoidal Dependence Model:**

    $$P_{release} = \frac{1}{1 + e^{-( [Ca^{2+}]_i - K_d ) / s}}$$

    where $K_d$ is the dissociation constant, $s$ is the steepness parameter.

The number of released neurotransmitters $N_{trans}$ can be modeled as a random variable, for example, using a binomial distribution:

$$P(N_{trans} = k) = \binom{N_{ves}}{k} P_{release}^k (1 - P_{release})^{N_{ves} - k}$$

where $N_{ves}$ is the number of ready-to-release vesicles.

**Connection with the Postsynaptic Neuron**

The released neurotransmitter binds to receptors on the postsynaptic membrane, causing a postsynaptic current, which was discussed in the section on dendrites. Thus, the axon model connects to the dendrite model through synaptic transmission.

**Factors Influencing the Speed of Action Potential Conduction**

The speed of action potential conduction depends on several factors that can be reflected in mathematical models:

*   **Axon Diameter:** Increasing the axon diameter reduces axial resistance, increasing conduction speed.
*   **Myelination:** The presence and thickness of the myelin sheath significantly increase speed through saltatory conduction.
*   **Density of Ion Channels:** The density of voltage-gated sodium channels at the nodes of Ranvier affects the efficiency of action potential generation.
*   **Temperature:** Temperature affects the kinetics of ion channels.

**Conclusion**

The mathematical formalization of the axon includes modeling the generation of the action potential based on ion channel dynamics, describing the propagation of the signal along the axon with consideration of myelination, and modeling the process of synaptic transmission. The Hodgkin-Huxley model is fundamental, but there are simpler models for specific tasks. Understanding the mathematical foundations of axon function is crucial for studying neural networks and developing neuro-inspired technologies.

**Limitations and Future Directions:**

*   **Heterogeneity:** There is significant heterogeneity in the properties of axons of different neuron types.
*   **Plasticity:** The properties of the axon, including channel conductance and myelination, can change over time.
*   **Molecular Details:** Models can be expanded to include more detailed molecular mechanisms of synaptic transmission.

## Synapses: Site of Intercellular Signal Transmission

Synapses are specialized structures that provide the transmission of information from one neuron to another or to effector cells (muscle or gland cells). There are two main types of synapses: chemical and electrical.

**Internal Structure of Synapses:**
*   **Chemical Synapses:** The most common type of synapse. Signal transmission occurs via chemical messengers - neurotransmitters.
    *   **Presynaptic Membrane:** The membrane of the axon terminal that releases neurotransmitters. Contains voltage-gated calcium channels and mechanisms for synaptic vesicle exocytosis.
    *   **Synaptic Cleft:** A narrow space (approximately 20-40 nm) between the presynaptic and postsynaptic membranes.
    *   **Postsynaptic Membrane:** The membrane of the target cell (usually the dendrite or soma of another neuron), containing receptors for neurotransmitters.
    *   **Neurotransmitters:** Chemical substances synthesized in the neuron and stored in synaptic vesicles. Examples: glutamate (the main excitatory neurotransmitter), GABA (the main inhibitory neurotransmitter), acetylcholine, dopamine, serotonin, norepinephrine, etc. Different neurotransmitters have different effects on the postsynaptic cell.
    *   **Receptors:** Protein molecules on the postsynaptic membrane that have specific affinity for certain neurotransmitters. Binding of neurotransmitter to the receptor causes a conformational change in the receptor and opens or closes ion channels, causing a change in the postsynaptic membrane potential (EPSP or IPSP). There are two main types of receptors:
        *   **Ionotropic Receptors:** Ligand-gated ion channels. Binding of neurotransmitter directly opens an ion channel, causing a fast and brief effect.
        *   **Metabotropic Receptors:** Coupled to G-proteins. Binding of neurotransmitter activates a G-protein, which in turn can either directly affect ion channels or activate second messengers, causing a slower and more prolonged effect.
    *   **Neurotransmitter Inactivation:** To prevent constant stimulation of the postsynaptic cell, neurotransmitters must be rapidly removed from the synaptic cleft. This can occur through:
        *   **Reuptake:** Transporter proteins on the presynaptic membrane or glial cells actively take up neurotransmitters back into the presynaptic terminal or glial cells.
        *   **Enzymatic Degradation:** Enzymes in the synaptic cleft break down neurotransmitters into inactive components.
*   **Electrical Synapses:** Less common than chemical synapses. Characterized by direct electrical connection between the presynaptic and postsynaptic cells through gap junctions.
    *   **Gap Junctions:** Formed by connexin proteins, which connect the cytoplasm of two adjacent cells, allowing ions and small molecules to freely move between them.
    *   **Fast Signal Transmission:** Signal transmission occurs almost instantaneously, without the delay characteristic of chemical synapses.
    *   **Bidirectional Transmission:** Signals can be transmitted in both directions.
    *   **Synchronization of Activity:** Electrical synapses play an important role in synchronizing the activity of groups of neurons.

Understanding the structure of the biological neuron and the mechanisms of signal transmission at the synapse is fundamental to studying the functioning of the nervous system, developing methods for treating neurological and psychiatric disorders, and creating more sophisticated artificial neural networks. Further research in this area continues to reveal the complex details of the functioning of these remarkable cells.

### **Synapses - Mathematical Model:**

### Chemical Synapses: Mathematical Formalization

The transmission of signals at a chemical synapse represents a complex cascade of events that can be broken down into several stages, each of which can be mathematically modeled.

**Presynaptic Processes: Arrival of the Action Potential and Calcium Influx**

The arrival of the action potential at the presynaptic terminal causes membrane depolarization, leading to the opening of voltage-gated calcium channels. The dynamics of the calcium current $I_{Ca}$ can be described similarly to the Hodgkin-Huxley model:

$$I_{Ca} = P_{Ca} g_{Ca} s^p (V_{pre} - E_{Ca})$$

where:
*   $P_{Ca}$ - the maximum permeability of calcium channels.
*   $g_{Ca}$ - the conductance of a single calcium channel.
*   $s$ - the activation variable of calcium channels, obeying a first-order differential equation.
*   $p$ - the Hill coefficient, reflecting the cooperativity of channel opening.
*   $V_{pre}$ - the presynaptic membrane potential.
*   $E_{Ca}$ - the equilibrium potential for calcium ions.

The dynamics of intracellular calcium concentration $[Ca^{2+}]_{pre}$ in the presynaptic terminal are described by the equation:

$$\frac{d[Ca^{2+}]_{pre}}{dt} = - \alpha I_{Ca} - \frac{[Ca^{2+}]_{pre} - [Ca^{2+}]_{rest}}{\tau_{Ca}}$$

where:
*   $\alpha$ - a constant linking the calcium current to the change in concentration.
*   $[Ca^{2+}]_{rest}$ - the resting calcium concentration.
*   $\tau_{Ca}$ - the effective time constant for calcium buffering and removal.

**Neurotransmitter Release: Probabilistic Models**

Neurotransmitter release is a quantal and probabilistic process. The probability of release $P_{release}$ depends on the concentration of calcium in the presynaptic terminal. A more accurate model than a simple sigmoidal function takes into account the number of calcium binding sites required to trigger vesicle fusion:

$$P_{release} = \frac{[Ca^{2+}]_{pre}^n}{K_d^n + [Ca^{2+}]_{pre}^n}$$

where $n$ is the number of calcium ions required to trigger release, and $K_d$ is the dissociation constant.

The number of released vesicles can be modeled as a random variable, following a binomial distribution (as previously mentioned) or a Poisson distribution if the release probability is small and the number of potential release sites is large.

The dynamics of the pool of ready-to-release vesicles can be described by a differential equation:

$$\frac{dN_{ves}}{dt} = -k_{release} P_{release} N_{ves} + k_{refill} (N_{max} - N_{ves})$$

where:
*   $N_{ves}$ - the number of ready-to-release vesicles.
*   $k_{release}$ - the release rate constant.
*   $k_{refill}$ - the refill rate constant.
*   $N_{max}$ - the maximum capacity of the vesicle pool.

**Diffusion and Binding of Neurotransmitter in the Synaptic Cleft**

The dynamics of neurotransmitter concentration $[NT]$ in the synaptic cleft can be described by a reaction-diffusion equation:

$$\frac{\partial [NT](x,t)}{\partial t} = D \nabla^2 [NT](x,t) - k_{bind} [NT](x,t) (R_{max} - [R_{bound}](x,t)) + k_{unbind} [R_{bound}](x,t) - k_{decay} [NT](x,t)$$

where:
*   $D$ - the diffusion coefficient of the neurotransmitter.
*   $\nabla^2$ - the Laplacian operator.
*   $k_{bind}$ - the binding rate constant of the neurotransmitter to the receptor.
*   $R_{max}$ - the total concentration of receptors.
*   $[R_{bound}]$ - the concentration of bound receptors.
*   $k_{unbind}$ - the unbinding rate constant of the neurotransmitter from the receptor.
*   $k_{decay}$ - the decay rate constant of the neurotransmitter.

In most cases, for simplification of modeling, it is assumed that the neurotransmitter is uniformly distributed in the cleft, and the equation reduces to an ordinary differential equation (as in the provided text).

**Postsynaptic Response: Ionotropic and Metabotropic Receptors**

**Ionotropic Receptors:** Binding of neurotransmitter to an ionotropic receptor causes the rapid opening of an ion channel. The current through the ionotropic receptor $I_{iono}$ can be modeled as:

$$I_{iono} = g_{iono} P_{open} (V_{post} - E_{rev})$$

where:
*   $g_{iono}$ - the maximum conductance of the channel.
*   $P_{open}$ - the probability of the channel being open, which depends on the concentration of bound neurotransmitter. For a single-channel model: $P_{open} = \frac{[NT]}{K_d + [NT]}$. For multi-channel models, the dependence may be more complex.
*   $V_{post}$ - the postsynaptic membrane potential.
*   $E_{rev}$ - the reversal potential of the ion channel.

**Metabotropic Receptors:** Binding of neurotransmitter to a metabotropic receptor activates a G-protein, which triggers a cascade of intracellular events. Modeling this process can be complex and includes equations for the concentration of G-proteins, second messengers, and the activity of kinases. In a simplified case, the effect of metabotropic receptors on ion channel conductance can be modeled as:

$$g_{metabotropic}(t) = g_{max} \cdot f([NT](t))$$

where $f([NT](t))$ is a function describing the dependence of conductance on neurotransmitter concentration, for example, a sigmoidal function. More detailed models account for the kinetics of G-protein activation and second messengers.

**Neurotransmitter Inactivation**

Processes of reuptake and enzymatic degradation affect the constant $k_{decay}$ in the neurotransmitter concentration equation. Reuptake can be modeled using the Michaelis-Menten equation for the transport rate:

$$v_{uptake} = V_{max} \frac{[NT]}{K_m + [NT]}$$

where $V_{max}$ is the maximum transport rate, and $K_m$ is the Michaelis-Menten constant. Enzymatic degradation is typically modeled as a first-order process.

### Electrical Synapses: Mathematical Formalization

Signal transmission in electrical synapses occurs through gap junctions, providing a direct electrical connection between cells. The current through the gap junction $I_{gap}$ between cells $i$ and $j$ can be described as:

$$I_{gap, ij} = g_{gap, ij} (V_i - V_j)$$

where:
*   $g_{gap, ij}$ - the conductance of the gap junction between cells $i$ and $j$.
*   $V_i$ and $V_j$ - the membrane potentials of cells $i$ and $j$, respectively.

The conductance of the gap junction $g_{gap, ij}$ may be constant or depend on various factors, such as pH, calcium concentration, and membrane potential. Modeling the dynamics of gap junction conductance may include equations describing the opening and closing of connexons forming the gap junction.

### Factors Influencing Synaptic Strength and Plasticity

The synaptic strength, which determines the magnitude of the postsynaptic response to a presynaptic signal, depends on multiple factors that can change during synaptic plasticity:

*   **Presynaptic Factors:** The probability of neurotransmitter release, the number of ready-to-release vesicles, the concentration of calcium in the presynaptic terminal.
*   **Postsynaptic Factors:** The number and type of receptors, the affinity of receptors for neurotransmitters, the efficiency of intracellular signaling cascades.
*   **Synaptic Cleft Properties:** The speed of diffusion and removal of neurotransmitters.

Mathematical models of synaptic plasticity describe the dynamics of these parameters depending on the activity of the synapse (e.g., the frequency of presynaptic spikes). Examples include models describing long-term potentiation (LTP) and long-term depression (LTD) based on changes in the number and efficiency of AMPA receptors.

### Conclusion

The mathematical formalization of synapses requires consideration of multiple interacting processes, from the arrival of the presynaptic action potential to the generation of the postsynaptic potential. The choice of specific models and the level of detail depend on the research question and available computational resources. Understanding the mathematical foundations of synaptic transmission is key to studying neural network function, mechanisms of learning and memory, and developing new therapeutic approaches for neurological and psychiatric disorders.

**Limitations and Future Directions:**

*   **Complexity and Heterogeneity:** Synapses demonstrate significant complexity and heterogeneity in their structure and function.
*   **Stochasticity:** Many processes at the synaptic level are stochastic, requiring the use of probabilistic models.
*   **Integration of Scales:** Linking models at the molecular level (e.g., receptor kinetics) with models at the cellular and network levels remains a complex challenge.

# Artificial Neuron

## Introduction

Artificial neural networks (ANNs) represent a class of computational models inspired by the structure and principles of biological neural networks. Historically, the interest in creating artificial intelligence led to the development of mathematical models capable of mimicking cognitive functions. ANNs, arising as a result of these studies, are a powerful tool for solving a wide range of problems, from pattern recognition to natural language processing. It is important to understand that ANNs are not an exact copy of biological systems but rather a mathematical abstraction of them. The purpose of this section is to provide a detailed examination of the structure and functioning of the artificial neuron, which is the fundamental building block of any artificial neural network. Understanding its components and mechanisms is essential for further comparison with biological analogs and for comprehending the capabilities and limitations of modern neural network technologies.

## Artificial Neuron and Its Composition

### Structure of the Artificial Neuron

The artificial neuron, in its basic form, represents a mathematical function that processes input signals and generates an output signal. To understand its operation, let's consider its main components and draw parallels with the biological neuron:

1. **Inputs (Inputs):** The artificial neuron receives a number of signals, mathematically represented as numerical values ($x_1, x_2, ..., x_n$). These inputs are analogous to the dendrites of a biological neuron, receiving signals from other neurons.

2. **Weights (Weights):** Each input signal is associated with a weight ($w_1, w_2, ..., w_n$). Weights are numerical coefficients that determine the strength or importance of the corresponding input signal. Similar to the synaptic strength in a biological neuron, weights define how much influence each input signal will have on the activation of the artificial neuron.

3. **Aggregation Function (Aggregation Function):** The received input signals, weighted by their respective weights, are combined using an aggregation function. The most common function is the simple summation of weighted inputs. This is analogous to the integration of incoming signals in the soma of a biological neuron.

4. **Activation Function (Activation Function):** The result of the aggregation function is passed to the activation function. This function performs a nonlinear transformation of the aggregated signal, determining whether the artificial neuron will be "activated" and what will be the intensity of its output signal. The activation function plays a key role in the ability of ANNs to model complex nonlinear dependencies. There are many types of activation functions, each with its own properties. For example, in biological neural networks, the activation function is typically an abstraction representing the firing rate of the neuron's action potential [4]. The resulting data is usually in the range [-1;1] or [0;1], and to determine the level of neuron activation, different types of activation functions are used (Fig. 2).

5. **Output (Output):** The output signal of the artificial neuron ($y$) is the result of applying the activation function ($\phi$) to the aggregated signal. This signal ($y$) is then passed to the inputs of other neurons in the network, analogous to how the axon of a biological neuron transmits a signal to other cells.

### Concept of the Artificial Neuron

The artificial neuron is a node in an artificial neural network, serving as a simplified model of a biological neuron [3]. A set of data is input into the artificial neural network, with each data element having its own input. Each input is weighted, i.e., multiplied by a certain coefficient. Then, each product is summed, and the resulting level of neuron activation is obtained. The block where the sum of all input values and weights is located corresponds to the cell body of the neuron (Fig. 2).

Thus, the artificial neuron is a key element of the artificial neural network, combining both mathematical principles of data processing and biological analogies, allowing the creation of complex and effective models for solving various tasks.

![Structure of the artificial neuron](https://neerc.ifmo.ru/wiki/images/a/a5/%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD_%D1%81%D1%85%D0%B5%D0%BC%D0%B0.png)

Fig. 2

### Inputs and Weights: Modeling Synaptic Connections

Input signals of the artificial neuron ($x_i$) represent numerical values that come from other neurons or the external environment. Weights ($w_i$) are key parameters of the artificial neuron that determine the strength of the connection between the current neuron and the source of the input signal. A large absolute value of the weight indicates a strong influence of the corresponding input on the activation of the neuron, while a weight close to zero indicates a weak influence. Negative weights can model inhibitory connections.

In addition to weights, the concept of **bias (bias)** ($b$) is often used. The bias represents an additional parameter that is added to the weighted sum of inputs. Its role is to shift the activation function, allowing the neuron to activate even when the input signals are zero, or conversely, to prevent activation when the inputs are non-zero. The bias can be considered analogous to the threshold of excitation in a biological neuron.

Mathematically, the weighted sum of inputs, taking into account the bias, is written as:

$$\sum_{i=1}^{n} w_i x_i + b$$

where:
*   $x_i$ - the value of the i-th input signal
*   $w_i$ - the weight associated with the i-th input signal
*   $n$ - the total number of input signals
*   $b$ - the value of the bias

### Aggregation Function: Summing Incoming Signals

The most common aggregation function in artificial neurons is the **weighted sum**, as shown above. This function simply sums the products of each input signal with its corresponding weight, and then adds the bias. The result of this operation is the **net input** (or simply *net*) of the neuron.

$$net = \sum_{i=1}^{n} w_i x_i + b$$

Although the weighted sum is the most common, other possible aggregation functions exist. For example, in some specialized architectures, functions based on the distance between input vectors and weights may be used, but they are much less common in standard models.

### Activation Function: Introducing Nonlinearity

The activation function plays a critical role in the operation of the artificial neuron by introducing **nonlinearity** into the model. Without nonlinearity, a multi-layer neural network is equivalent to a single-layer perceptron, as successive linear transformations can be reduced to a single linear transformation. Nonlinearity allows neural networks to approximate complex, nonlinear dependencies in data.

Let's consider the most common types of activation functions:

*   **Threshold Function (Threshold function):** This is the simplest activation function, which outputs 1 if the input exceeds a certain threshold, and 0 otherwise. It is analogous to the "all-or-nothing" principle when generating an action potential in a biological neuron. Mathematically, it can be represented as:

    $$f(x) = \begin{cases} 1, & \text{if } x \ge \theta \\ 0, & \text{if } x < \theta \end{cases}$$

    where $\theta$ is the threshold value.

*   **Sigmoid Function (Sigmoid function):** The sigmoid function, also known as the logistic function, is a smooth, differentiable function that maps the input value to the range from 0 to 1. Historically, it was very popular because its output could be interpreted as a probability. The mathematical notation is:

    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

*   **Hyperbolic Tangent (Tanh function):** The hyperbolic tangent function is similar to the sigmoid function but its range of values is from -1 to 1. This can be useful in some architectures where data centering is required. The mathematical notation is:

    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

*   **ReLU Function (Rectified Linear Unit):** ReLU is one of the most popular activation functions in modern deep neural networks due to its simplicity and efficiency in training. It returns the input value if it is positive, and 0 otherwise. The mathematical notation is:

    $$f(x) = \max(0, x)$$

    There are various variants of ReLU, such as **Leaky ReLU** and **ELU (Exponential Linear Unit)**, which introduce a small slope for negative input values to avoid the "dead neuron" problem, where a neuron stops activating.

    *   **Leaky ReLU:** $f(x) = \begin{cases} x, & \text{if } x > 0 \\ \alpha x, & \text{if } x \le 0 \end{cases}$, where $\alpha$ is a small constant (e.g., 0.01).
    *   **ELU:** $f(x) = \begin{cases} x, & \text{if } x > 0 \\ \alpha (e^x - 1), & \text{if } x \le 0 \end{cases}$, where $\alpha$ is a positive constant.

*   **Other Activation Functions:** There are many other activation functions developed to solve specific tasks, such as **Softmax**, which is used in output layers for multi-class classification tasks, and **Swish**, which has shown good results in some architectures.

The choice of activation function is an important aspect of designing a neural network and depends on the specific task and network architecture. Different activation functions have different properties, such as differentiability, value range, computational complexity, and ability to prevent gradient vanishing or explosion during training.

### Output of the Artificial Neuron

The output signal of the artificial neuron ($y$) is the result of applying the activation function ($\phi$) to the aggregated signal. Mathematically, this can be written as:

$$y = \phi(net) = \phi(\sum_{i=1}^{n} w_i x_i + b)$$

This output signal is then passed to the inputs of other neurons in subsequent layers. Depending on the task, the output signal may represent a classification, a regression value, or other types of data. In multi-layer neural networks, the output of one neuron becomes the input for neurons in the next layer, forming a complex network of interconnections.

## Mathematical Model of the Artificial Neuron

All the work of the artificial neuron can be reduced to a single mathematical equation that describes the transformation of input signals into the output signal:

$$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$

where:
*   $y$ - the output signal of the neuron
*   $\phi$ - the activation function
*   $w_i$ - the weights of the input signals
*   $x_i$ - the input signals
*   $b$ - the bias
*   $n$ - the number of input signals

During the training of a neural network, it is precisely the parameters of the model – **weights ($w_i$) and bias ($b$)** – that are adjusted so that the network can perform the assigned task, for example, correctly classify images or predict values. The activation function ($\phi$) is usually chosen in advance and remains unchanged during training.

## Conclusion

Despite its apparent simplicity, the artificial neuron is a powerful building block for creating complex artificial neural networks. It represents a mathematical model that imitates certain aspects of the biological neuron, such as receiving inputs, weighting them, aggregating them, and generating an output signal. It is important to understand that the artificial neuron is a **simplified model** of the biological neuron, ignoring many complex biochemical and physiological processes that occur in real neural cells. Nevertheless, this simplified model has proven to be extremely effective for solving a wide range of problems.

In the next part of our work, we will conduct a detailed comparison of the structure and functioning of biological and artificial neurons, identifying both similarities and fundamental differences between these systems.

# Comparative Analysis of Biological and Artificial Neurons

We will conduct a detailed comparative analysis of biological and artificial neurons, examining their structural, functional, and computational aspects.

1. **Structural Comparison:**

**Biological Neuron:**
- **Soma:** A complex cellular structure containing the nucleus with genetic material and various organelles responsible for metabolic processes and synthesis of necessary molecules. The cytoskeleton provides structural support and participates in substance transport.
- **Dendrites:** Branched processes specialized in receiving incoming signals from other neurons through synapses. The morphology of dendrites (number, length, branching) significantly affects signal integration. Dendritic spines increase the surface area for synaptic contacts and possess plasticity.
- **Axon:** A single long process designed for transmitting outgoing signals in the form of action potentials. The initial segment of the axon (axon hillock) plays a key role in initiating the action potential. The axon may be covered by a myelin sheath, providing fast saltatory conduction.
- **Synapses:** Specialized structures that transmit signals between neurons. Chemical synapses use neurotransmitters released by the presynaptic neuron and bind to receptors on the postsynaptic neuron, causing excitatory or inhibitory postsynaptic potentials. Electrical synapses (gap junctions) also exist, providing direct and fast ionic connections between neurons.

**Artificial Neuron:**
- **Inputs:** Represent numerical values corresponding to the activity of previous neurons or input data. The number of inputs determines the dimensionality of the input vector.
- **Weights:** Numerical parameters modeling the strength of synaptic connections. Positive weights correspond to excitatory connections, negative ones to inhibitory connections. During training, weights are adjusted to optimize network performance.
- **Aggregation Function:** Usually represents a weighted sum of input signals. May include a bias (bias), analogous to the neuron's threshold of excitation.
- **Activation Function:** A nonlinear function applied to the result of aggregation to determine the output signal of the neuron. Various activation functions (sigmoid, ReLU, tanh, etc.) introduce nonlinearity, allowing the model to approximate complex dependencies.
- **Output:** A numerical value representing the neuron's activity and passed to the inputs of other neurons.

2. **Functional Comparison:**

**Biological Neuron:**
- **Electrochemical Signals:** Information transmission is based on changes in membrane potential caused by ion flow through ion channels.
- **Complex System of Ion Channels:** Various types of voltage-dependent and ligand-dependent ion channels ensure the generation and propagation of the action potential, as well as modulate synaptic transmission.
- **Nonlinear Signal Transmission:** Nonlinearity arises at various levels, including the nonlinear behavior of ion channels, synaptic transmission saturation, and dendritic integration.
- **Adaptive Synaptic Plasticity:** Synaptic strength can change over time depending on neuronal activity (synaptic plasticity, e.g., long-term potentiation and depression), which is the basis of learning and memory. There are various forms of plasticity depending on time and mechanisms.
- **Continuous Operation in Real-Time:** Biological neurons operate asynchronously and continuously, processing information in real-time.

**Artificial Neuron:**
- **Numerical Values:** Information processing is carried out by manipulating numerical values.
- **Mathematical Operations of Weighting and Summation:** The main mechanism for processing input signals.
- **Nonlinear Activation Functions:** Introduce nonlinearity, necessary for modeling complex functions. The choice of activation function affects the network's learnability and performance.
- **Weight Adjustment During Training:** Training occurs through iterative weight adjustment using optimization algorithms (e.g., gradient descent) based on training data.
- **Discrete Data Processing:** In most implementations, artificial neural networks process data discretely, although there are models that simulate continuous time.

**Mathematical Models:**

**Biological Neuron:**
- **Hodgkin-Huxley Model for Action Potential:**
  $$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$
  where $C_m$ is membrane capacitance, $V_m$ is membrane potential, $t$ is time, $I_{ion}$ is total ionic current (including sodium, potassium, and leakage currents), $I_{ext}$ is external applied current. This model describes the dynamics of membrane potential based on the behavior of individual ion channels and is fundamental to understanding action potential generation. There are other, simpler models, such as the FitzHugh-Nagumo model, which retain the main dynamic properties but with less computational complexity.

**Artificial Neuron:**
- **Simple Mathematical Model:**
  $$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$
  where $y$ is the neuron's output, $\phi$ is the activation function, $w_i$ is the weight of the $i$-th input, $x_i$ is the value of the $i$-th input, $b$ is the bias. This model represents a significant simplification of the biological neuron, focusing on the basic principles of weighted summation and nonlinear transformation. There are more complex artificial neuron models, such as recurrent neurons, which include feedback loops and can process sequential data, or convolutional neurons, specializing in processing spatially organized data.

**Key Conclusions of Comparative Analysis:**

1. **Main Similarities:**
- **Functional Similarity in Signal Processing:** Both types of neurons perform the fundamental function of receiving, processing, and transmitting information. Biological neurons integrate electrochemical signals, while artificial ones process numerical values.
- **Principle of Input Integration:** Both biological and artificial neurons integrate incoming signals, although the summation mechanisms differ (spatial-temporal integration in dendrites versus weighted summation).
- **Presence of Nonlinearity:** Nonlinearity is a key property of both types of neurons, allowing them to model complex dependencies. In biological neurons, nonlinearity is due to the properties of ion channels and synaptic transmission, while in artificial ones, it is due to activation functions.
- **Ability to Learn and Adapt:** Both biological and artificial neural networks can change their parameters (synaptic strength or weights) in response to experience, allowing them to adapt to new tasks and data. However, the learning mechanisms differ fundamentally.

2. **Fundamental Differences:**
- **Information Processing Substrate:** The biological neuron is based on complex biochemical and electrophysiological processes involving ion movement, neurotransmitters, and membrane potentials. The artificial neuron operates on abstract mathematical operations on numbers.
- **Structural and Dynamic Complexity:** Biological neurons demonstrate much higher structural complexity at the molecular, cellular, and network levels. Their dynamics include numerous interacting processes that are not yet fully reproduced in artificial models.
- **Temporal Scales and Synchronicity:** Biological neurons operate on millisecond timescales and can exhibit complex synchronous activity patterns. Artificial neural networks often operate in discrete time, although research in spiking neural networks aims for more realistic modeling of temporal dynamics.
- **Learning and Adaptation Mechanisms:** Learning in biological systems involves complex biochemical processes, such as synaptic plasticity, modulated by various neurotransmitters. Learning in artificial neural networks is typically implemented using optimization algorithms, such as backpropagation.
- **Energy Efficiency:** Biological neurons demonstrate remarkable energy efficiency compared to modern computing systems implementing artificial neural networks.

3. **Advantages and Limitations:**

**Biological Neuron:**
+ **High Adaptability and Learnability:** Biological neural networks have exceptional ability to adapt to new conditions and learn from a limited number of examples.
+ **Energy Efficiency:** Operates with extremely low power consumption, which is a subject of active research in neuromorphic computing.
+ **Parallel and Distributed Information Processing:** Massive parallelism and distribution ensure robustness to damage and high efficiency.

- **Relatively Slow Signal Transmission Speed:** The speed of action potential propagation is limited by physiological factors.
- **Susceptibility to Fatigue and Biological Limitations:** Neuron function depends on metabolic processes and can be disrupted by various factors.

**Artificial Neuron:**
+ **High Computation Speed:** Can perform mathematical operations significantly faster than biological neurons.
+ **Simplicity of Scaling and Reproducibility:** The architecture and parameters of artificial neural networks can be easily scaled and reproduced.
+ **Stability and Predictability of Operation:** With proper configuration, artificial neural networks demonstrate stable operation.

- **Simplified Information Processing Model:** Does not reflect the full complexity of biological neurons, limiting its capabilities in some areas.
- **High Energy Consumption:** Training and operation of large artificial neural networks require significant computational resources and energy.
- **Limited Generalization and Adaptation Ability:** May struggle with generalizing to new, unseen data and requires a large number of training examples.

This detailed analysis demonstrates that the artificial neuron, despite its simplicity compared to the biological prototype, successfully imitates the core principles of information processing, enabling it to effectively solve a wide range of tasks in the field of artificial intelligence. However, it is important to understand the fundamental differences and limitations of artificial models and to continue research aimed at creating more biologically realistic neural networks. Future research directions include the development of neuromorphic chips that mimic the architecture and dynamics of biological neural networks.

# Learning in Biological and Artificial Neural Networks

## Introduction

Learning is a fundamental process underlying the adaptation and functioning of both biological and artificial neural networks. This ability allows neural networks to modify their internal parameters—synaptic weights in biological systems and connection weights in artificial neural networks (ANNs)—to optimize task performance. In this section, we will examine the key principles of learning in both types of networks, with a particular focus on Hebb's rule and its derivatives, which play a significant role in understanding learning and memory mechanisms.

## Learning in Biological Neural Networks: Synaptic Plasticity and Hebb's Rule

### Synaptic Plasticity as the Basis of Learning

Synaptic plasticity, the fundamental ability of synapses to change their strength or transmission efficiency over time, is the cornerstone of learning and memory in biological neural networks. It is through this dynamic adaptation that neural circuits can modify their responses to external stimuli and internal states. The two main manifestations of synaptic plasticity, directly underlying learning processes, are long-term potentiation (LTP) and long-term depression (LTD). LTP represents a sustained strengthening of synaptic transmission, reflecting processes of memory formation and association, whereas LTD leads to its weakening, which may be related to forgetting or adapting to changes in the environment.

### **Long-Term Potentiation (LTP)**

![Long-Term Potentiation (LTP)](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-05/assets/%D0%A0%D0%B8%D1%81%D1%83%D0%BD%D0%BE%D0%BA_N.jpg)

**Long-Term Potentiation (LTP)** is a biochemical process where repeated synchronous activation of two connected neurons leads to a sustained strengthening of synaptic transmission. The mechanism of LTP is best studied in the hippocampus, a brain structure critical for memory formation.

Let's imagine a synapse as a kind of **signal amplifier** between two neurons. Usually, when the presynaptic neuron sends a signal, the postsynaptic neuron receives an amplified version of that signal. LTP is the process that **increases the amplification factor** of this "amplifier" for a long time. This means that the same input signal from the presynaptic neuron will now cause a **stronger response** in the postsynaptic neuron.

Now let's break down the **mechanism of LTP step by step**, as if it were an algorithm the cell uses to strengthen the synaptic connection:

1.  **Presynaptic Neuron Activity and Glutamate Release:** Imagine the presynaptic neuron "activates" – as if it "decided" to send a message. To do this, it releases a chemical "signal" – **glutamate**. Glutamate is like a "key-signal".
2.  **Glutamate Binding to AMPA Receptors and Depolarization:** On the postsynaptic neuron, there are "receptors" for glutamate – **AMPA receptors**. When glutamate "binds" to AMPA receptors, it opens ion channels, allowing sodium ions (Na⁺) to enter the cell. This causes **depolarization** of the postsynaptic neuron's membrane. Depolarization is like changing the cell's electrical potential, making it more "excitable".
3.  **Removal of Magnesium Blockade of NMDA Receptors:** On the postsynaptic membrane, there are other "receptors" – **NMDA receptors**. Under normal conditions, these receptors are "blocked" by magnesium ions (Mg²⁺). However, when **depolarization** occurs (from step 2), this blockade is lifted. Imagine depolarization as a "key" that temporarily "opens" the NMDA receptors, but not completely.
4.  **Calcium Ion (Ca²⁺) Influx through NMDA Receptors:** Now, with the NMDA receptors "unblocked" and glutamate binding (as with AMPA receptors), they become permeable to **calcium ions (Ca²⁺)**. Calcium ions enter the postsynaptic cell. **This is the key moment of LTP!** Calcium is like a "signaling ion" that triggers a whole cascade of intracellular events.
5.  **Activation of Intracellular Kinases (e.g., CaMKII) and Long-Term Changes:** Inside the cell, calcium ions activate specific proteins – **kinases**, such as CaMKII (calcium-calmodulin-dependent protein kinase II). Kinases are like "switch enzymes" that trigger other processes. CaMKII and other kinases do two important things:
    *   **Increase the number of AMPA receptors on the membrane:** This is like adding more "receptors" for glutamate. Now, when glutamate is released next time, more AMPA receptors can "catch" it, leading to a stronger depolarization.
    *   **Initiate the synthesis of new synaptic proteins and structural changes:** This is a longer-term effect. Kinases initiate processes that lead to physical changes in the synapse, such as the growth of dendritic spines (protrusions on dendrites where synapses are located). This is like physically "reinforcing" the connection between neurons.

**Conclusion:**

As a result of these steps, the synapse becomes **more efficient** in signal transmission. Now, when the presynaptic neuron activates, the postsynaptic neuron will respond **stronger and faster**. And most importantly, this strengthening of the synaptic connection **persists for a long time** – hence the name "long-term potentiation".

**Mathematical Analogy:**

Imagine a function $f(x)$ describing the strength of a synaptic connection, where $x$ is the input signal from the presynaptic neuron, and $f(x)$ is the output signal of the postsynaptic neuron. LTP is a process that modifies the parameters of this function so that for the same input signal $x$, the output $f(x)$ becomes larger. LTP changes the "weight" of the synaptic connection, similar to how weights are adjusted in artificial neural networks during training.

**Key Steps of LTP:**
   - During high activity of the **presynaptic neuron**, glutamate is released.
   - Glutamate binds to **AMPA receptors** on the postsynaptic neuron, causing its depolarization.
   - This removes the magnesium blockade of **NMDA receptors**, allowing calcium ions (Ca²⁺) to enter the postsynaptic cell.
   - The sudden influx of Ca²⁺ activates intracellular kinases (e.g., CaMKII), which:
     - Increase the number of AMPA receptors on the membrane.
     - Initiate the synthesis of new synaptic proteins, leading to structural changes in the synapse (e.g., growth of dendritic spines).

### **Long-Term Depression (LTD)**

**Long-Term Depression (LTD)** is the opposite process, weakening synaptic connections. It plays a role in "forgetting" irrelevant information and optimizing neural networks.

Recall the "signal amplifier" analogy. If LTP increases the "amplification factor," then **Long-Term Depression (LTD)** is the process that **decreases the amplification factor** of the synaptic "amplifier" for a long time. This means that the same input signal from the presynaptic neuron will now cause a **weaker response** in the postsynaptic neuron.

Now let's break down the **mechanism of LTD step by step**:

1.  **Weak or asynchronous activity of the presynaptic neuron and moderate glutamate release:** Unlike LTP, LTD requires **weak or irregular** activity of the presynaptic neuron. Imagine the neuron sending "quiet" or "irregular" messages. At the same time, **glutamate** is released, but in smaller amounts or in a different manner than in LTP.
2.  **Glutamate binding to AMPA and NMDA receptors and moderate depolarization:** Glutamate binds to both AMPA and NMDA receptors. AMPA receptors cause a **small depolarization** of the postsynaptic membrane. NMDA receptors are also activated, but due to the weak depolarization and other factors, **fewer calcium ions (Ca²⁺)** enter the postsynaptic cell than in LTP. **This is the key difference from LTP – the level of calcium!**
3.  **Moderate calcium ion (Ca²⁺) influx and activation of phosphatases:** The moderate level of Ca²⁺ activates another type of intracellular enzyme – **phosphatases**, such as calcineurin. Phosphatases act in the **opposite direction to kinases** – they do not add phosphate groups to proteins but **remove** them.
4.  **Phosphatase action and reduction of AMPA receptors:** Activated phosphatases, particularly calcineurin, initiate processes that lead to the **removal of AMPA receptors** from the synaptic membrane. This is like "removing" some of the "receptors" for glutamate. Now, when glutamate is released next time, fewer AMPA receptors can "catch" it, leading to a **weaker depolarization**.
5.  **Suppression of synaptic protein synthesis and reduction in synapse size (in some cases):** Phosphatases can also suppress the synthesis of new synaptic proteins and even contribute to the **reduction in synapse size**. This is like physically "weakening" or "reducing" the connection between neurons.

**Conclusion:**

As a result of these steps, the synapse becomes **less efficient** in signal transmission. Now, when the presynaptic neuron activates, the postsynaptic neuron will respond **weaker**. And this weakening of the synaptic connection also **persists for a long time** – hence the name "long-term depression".

**Mathematical Analogy:**

Recall the function $f(x)$ describing the strength of a synaptic connection. LTD is a process that modifies the parameters of this function so that for the same input signal $x$, the output $f(x)$ becomes **smaller**. LTD also changes the "weight" of the synaptic connection, but in the direction of **decrease**.

**Key Steps of LTD:**
   - During weak or asynchronous activity of the presynaptic neuron, the level of Ca²⁺ in the postsynaptic cell increases moderately.
   - This activates phosphatases (e.g., calcineurin), which:
     - Remove AMPA receptors from the membrane.
     - Suppress the synthesis of synaptic proteins.
   - In some cases, LTD is accompanied by a reduction in synapse size.

**Role of LTP and LTD in Learning:**
- **Associative Learning**: LTP implements the Hebbian principle ("neurons that fire together, wire together"). For example, if neuron A repeatedly activates neuron B at the moment of a significant stimulus (sound + food), their connection strengthens, forming a conditioned reflex.
- **Spatial Memory**: In the hippocampus, LTP encodes maps of locations, while LTD erases outdated data when the environment changes.
- **Neuroplasticity**: LTD eliminates redundant connections during critical periods of brain development (e.g., visual cortex maturation).

These processes are dynamically balanced: for example, during learning a new skill, LTP strengthens the "correct" connections, while LTD suppresses competing pathways. Disruption of the LTP/LTD balance is linked to neurodegenerative diseases (e.g., Alzheimer's disease) and psychiatric disorders.

### Hebb's Rule: "Neurons that fire together, wire together"

Hebb's rule, a fundamental principle formulated by Donald Hebb, states: "When the axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in its excitation, some growth or metabolic change occurs in one or both cells such that A's efficiency as a cause of B's firing is increased." In the context of synaptic plasticity, this rule is interpreted as the mechanism by which the connection between two neurons strengthens when they are activated simultaneously. Mathematically, Hebb's rule can be expressed as:

$$\Delta w_{ij}(t) = \eta \cdot y_i(t) \cdot y_j(t)$$

where:
*   $\Delta w_{ij}(t)$ – change in the strength of the synaptic connection between presynaptic neuron $j$ and postsynaptic neuron $i$ at time $t$.
*   $\eta$ – a positive constant called the **learning rate** or **learning coefficient**. It determines the magnitude of the weight change during each update.
*   $y_i(t)$ – activity (firing rate or depolarization level) of the postsynaptic neuron $i$ at time $t$.
*   $y_j(t)$ – activity (firing rate or depolarization level) of the presynaptic neuron $j$ at time $t$.

This equation formalizes the idea that the change in synaptic weight is proportional to the product of the activities of the presynaptic and postsynaptic neurons.

### Modern Learning Methods in ANNs: Gradient Descent and Backpropagation

In modern artificial neural networks (ANNs), both **supervised** and **unsupervised** learning methods are widely used. For training multi-layer neural networks, such as multi-layer perceptrons, the dominant algorithm is **backpropagation**.

Backpropagation is an algorithm based on **gradient descent**. The goal of training is to minimize the **loss function** $J(\mathbf{W})$, where $\mathbf{W}$ represents all the weights in the network. Weight updates occur in the direction opposite to the gradient of the loss function:

$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \nabla J(\mathbf{W}^{(t)})$

where $\eta$ is the learning rate.

The gradient of the loss function with respect to a specific weight $w_{ij}^{(l)}$ in layer $l$ is calculated using the chain rule, leading to the following weight update rule:

$\Delta w_{ij}^{(l)} = - \eta \cdot \delta_i^{(l)} \cdot a_j^{(l-1)}$

where $\delta_i^{(l)}$ is the local error for neuron $i$ in layer $l$, and $a_j^{(l-1)}$ is the activation of the neuron in the previous layer.

A key difference from direct application of Hebb's rule is that learning using backpropagation is **based on error**. Weight updates occur not simply based on the simultaneous activation of connected neurons, but on how much the network's prediction differs from the desired result. Furthermore, for supervised learning using backpropagation, **labeled data** is required, i.e., a set of inputs with corresponding correct answers.

### Relationship between Gradient Descent and Hebbian Learning Principles

Despite their differences, learning using backpropagation can be viewed as a more complex form of Hebbian learning. Weight updates in backpropagation also depend on the activity of connected neurons, albeit indirectly through the error gradient. Research is ongoing to develop local learning rules in ANNs that are more biologically plausible and scalable, while retaining the effectiveness of modern algorithms. These studies often aim to integrate Hebbian-like learning principles with error-based learning to create more powerful and flexible artificial intelligence systems.

### Conclusion

Artificial and biological neural networks, despite sharing the same goal of information processing, have significant structural, functional, and computational differences.

**Structural Differences:**
*   Biological neurons have a complex cellular structure, including soma, dendrites, axon, and synapses, enabling the transmission of electrochemical signals.
*   Artificial neurons represent mathematical functions that process numerical values through inputs, weights, aggregation functions, and activation functions.
*   In biological neurons, signal transmission occurs via ions and neurotransmitters, whereas in artificial neurons, mathematical operations on numbers are used.

**Functional Differences:**
*   Biological neurons use nonlinear signal transmission based on complex interactions of ion channels and synaptic plasticity.
*   Artificial neurons introduce nonlinearity using activation functions, allowing them to model complex dependencies.
*   Biological neurons learn through synaptic plasticity, including long-term potentiation (LTP) and long-term depression (LTD), as well as spike-timing-dependent plasticity (STDP), whereas artificial neural networks learn using optimization algorithms like gradient descent and backpropagation.
*   Biological neural networks operate continuously and asynchronously, while artificial neural networks often process data discretely.
*   In biological neurons, modulators such as dopamine, serotonin, and norepinephrine influence plasticity.

**Similarities:**
*   Both types of neurons perform the function of receiving, processing, and transmitting information.
*   Both integrate incoming signals, although the summation mechanisms differ.
*   Both possess nonlinearity, allowing them to model complex dependencies.
*   Both systems are capable of learning and adaptation, changing their parameters (synaptic strength or weights) in response to experience.
*  Both types of neural networks use **Hebb's rule** (or its extensions) for learning, where simultaneous activation of neurons leads to strengthening their connection.

**Mathematical Models:**
*   The dynamics of the membrane potential in biological neurons are described by the **Hodgkin-Huxley model**, which accounts for sodium, potassium, and leakage currents.
    *   $$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$
*   In contrast, the operation of an artificial neuron is described by a simple mathematical model:
    *   $$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$
where $y$ is the neuron's output, $\phi$ is the activation function, $w_i$ is the weight of the $i$-th input, $x_i$ is the value of the $i$-th input, and $b$ is the bias.

**Advantages and Limitations:**
*   Biological neurons exhibit high adaptability, energy efficiency, and parallel processing, but have relatively slow signal transmission speeds.
*   Artificial neurons offer high-speed computation, ease of scaling and reproducibility, but consume significant energy and have limited generalization capabilities.
*   **Key Takeaways:**
    *   The artificial neuron is a simplified model of the biological neuron, yet it successfully imitates the core principles of information processing.
    *   It is important to understand the fundamental differences and limitations of artificial models and strive to create more biologically realistic neural networks.
    *   Future research directions include the development of neuromorphic chips that mimic the architecture and dynamics of biological neural networks.

**:

*   **Long-term potentiation (LTP):** LTP is a vivid example of a cellular learning mechanism, where high-frequency stimulation of the presynaptic neuron induces a persistent enhancement of the postsynaptic neuron’s response. The key event in LTP induction is the activation of NMDA receptors on the postsynaptic membrane. Upon depolarization of the postsynaptic cell and binding of glutamate to NMDA receptors, the ion channel associated with the receptor opens, allowing calcium ions (Ca<sup>2+</sup>) to enter the cell. This influx of calcium triggers a cascade of intracellular signaling pathways that lead to long-lasting changes in synaptic strength.

*   **Long-term depression (LTD):** In contrast to LTP, long-term depression (LTD) is a process of synaptic weakening. LTD is often induced by low-frequency stimulation of presynaptic neurons. LTD mechanisms also involve changes in intracellular calcium concentration, but in different spatiotemporal patterns, leading to activation of distinct signaling pathways that cause receptor internalization and synaptic weakening.

*   **Molecular mechanisms underlying Hebbian learning:**

The molecular mechanisms underlying Hebbian learning involve a complex cascade of biochemical reactions triggered by the influx of calcium ions (Ca<sup>2+</sup>) into the postsynaptic cell through NMDA receptors. This Ca<sup>2+</sup> influx acts as a key second messenger, activating a range of calcium-dependent enzymes, including **calcium/calmodulin-dependent protein kinase II (CaMKII)**. CaMKII plays a central role in LTP by phosphorylating **AMPA receptors**. Phosphorylation of AMPA receptors increases their ionic conductance and promotes their insertion into the postsynaptic membrane, resulting in an increased number of functional receptors on the cell surface and consequently enhanced synaptic response. Other protein kinases, such as **protein kinase A (PKA)** and **protein kinase C (PKC)**, are also involved in regulating synaptic plasticity by participating in various signaling pathways that modulate gene expression and synaptic structure.

In LTD processes, **protein phosphatases**, such as **calcineurin**, play a critical role. Activation of phosphatases leads to dephosphorylation of AMPA receptors, reducing their conductance and promoting their internalization (removal) from the postsynaptic membrane, thereby weakening synaptic transmission. Importantly, maintaining long-term changes in synaptic strength—particularly during memory consolidation—requires **synthesis of new proteins**. Activation of specific signaling pathways, such as the MAPK/ERK pathway, leads to gene transcription and synthesis of proteins necessary for structural changes in synapses, ensuring the persistence of LTP and LTD effects.

**Extended description of molecular mechanisms:**

Molecular mechanisms of synaptic plasticity exhibit a complex **temporal scale**, unfolding from milliseconds to days or longer. Early phases of LTP (early LTP, E-LTP), lasting from several minutes to an hour, do not require new protein synthesis and rely on covalent modifications of existing proteins, such as phosphorylation of AMPA receptors. Late phases of LTP (late LTP, L-LTP), which support long-term memory, require **synthesis of new proteins** and changes in gene expression. This process involves activation of transcription factors such as CREB (cAMP-response element binding protein), which bind to DNA and initiate transcription of genes encoding structural proteins, growth factors, and other molecules necessary for stable synaptic changes.

**Spatial organization of signaling complexes** also plays a critical role. Signaling molecules are not diffusely distributed in the cell but are organized into specialized microdomains near synapses. For example, NMDA receptors, AMPA receptors, CaMKII, and other signaling proteins form the **postsynaptic density (PSD)**—a complex protein structure that enables efficient signal transmission and integration. Various scaffold and adapter proteins ensure precise localization and interaction of these molecules.

The **role of local protein synthesis** in synaptic plasticity is becoming increasingly evident. mRNA of specific proteins is transported into dendrites and even near synapses, where they can be locally translated into proteins in response to synaptic activity. This enables rapid and specific synaptic responses to stimulation without requiring protein transport from the cell body.

**Synaptic tagging mechanisms** explain how synaptic plasticity can be specific to particular synapses, even when stimulation affects multiple neurons. According to this concept, active synapses receive a molecular "tag" that allows them to capture newly synthesized proteins required for long-term changes. This tag may consist of transient biochemical changes that render the synapse receptive to plasticity consolidation factors.

## Multilevel regulation of synaptic plasticity

Synaptic plasticity is not an isolated process occurring at the level of a single synapse. It undergoes complex multilevel regulation involving interactions between neurons and glial cells, epigenetic mechanisms, and modulation by various neurotransmitters and neuromodulators.

### Diversity of forms of synaptic plasticity

In addition to long-term potentiation (LTP) and long-term depression (LTD)—the most extensively studied forms of synaptic plasticity—there exists a broad spectrum of other mechanisms modulating synaptic strength across various temporal scales. These include **short-term plasticity**, encompassing **synaptic facilitation** and **synaptic depression**. Synaptic facilitation manifests as a short-term (from milliseconds to seconds) increase in the amplitude of the postsynaptic potential in response to repeated presynaptic stimulation. This effect is caused by accumulation of residual calcium in the presynaptic terminal, increasing the probability of neurotransmitter release during subsequent impulses. In contrast, synaptic depression is characterized by a temporary reduction in synaptic transmission efficacy during high-frequency stimulation, potentially due to depletion of the neurotransmitter vesicle pool in the presynaptic terminal. **Homeostatic plasticity** represents another important class of mechanisms aimed at maintaining long-term stability of neuronal activity. It includes processes such as synaptic scaling, where all synaptic inputs to a neuron are uniformly strengthened or weakened, and regulation of neuronal excitability by altering the density of ion channels. **Metaplasticity**, as previously mentioned, modulates the very capacity of synapses for plasticity by changing the thresholds for LTP and LTD induction based on prior neuronal activity.

### Metaplasticity and its role in learning

Metaplasticity is plasticity of synaptic plasticity. It describes the ability of prior neural activity to alter the subsequent capacity of synapses for LTP or LTD. In other words, the history of neuronal activity influences its current ability to learn. For example, prior high-frequency stimulation may "prime" a synapse for easier induction of LTP in the future. Metaplasticity plays a crucial role in stabilizing synaptic changes and preventing saturation of plasticity. Molecular mechanisms of metaplasticity include changes in receptor expression, thresholds for LTP/LTD induction, and functioning of signaling cascades.

It is important to note that synaptic plasticity is not a rigidly fixed process but is finely regulated by various neurotransmitters and neuromodulators. Classical neurotransmitters such as glutamate and GABA not only mediate rapid synaptic transmission but also play modulatory roles in plasticity. For instance, activation of different types of glutamate receptors (NMDA, AMPA, mGluR) initiates distinct signaling cascades leading to LTP or LTD. GABAergic inhibition can also modulate plasticity by influencing the postsynaptic depolarization threshold required for LTP induction. Neuromodulators such as dopamine, norepinephrine, serotonin, and acetylcholine exert broader and longer-lasting effects on synaptic plasticity, altering the "rules of the game" for learning. Dopamine, for example, plays a key role in reinforcement learning, enhancing synaptic plasticity in neural circuits associated with reward acquisition. Norepinephrine, released during stress or novelty, can enhance memory consolidation by modulating synaptic plasticity in emotionally relevant brain regions such as the amygdala. Acetylcholine, critical for attention and learning, modulates plasticity in the hippocampus and cerebral cortex. Serotonin influences mood and emotional state, which can indirectly affect learning and memory processes. Moreover, neuropeptides such as opioid peptides and neuropeptide Y can also modulate synaptic plasticity by regulating emotional and motivational aspects of learning.

### Neuron-glia interactions in memory processes

Glial cells, particularly astrocytes and microglia, play an active role in modulating synaptic plasticity and learning processes. Astrocytes surrounding synapses can release gliotransmitters such as glutamate, D-serine, and ATP, which influence synaptic transmission and plasticity. They also participate in regulating extracellular concentrations of ions and neurotransmitters, maintaining optimal synaptic function. Microglia, the brain’s immune cells, can also affect synaptic plasticity, especially in the context of inflammation and neurodegenerative diseases, through phagocytosis of synapses and release of cytokines. In the healthy brain, microglia participate in synaptic pruning, which is essential for forming mature neural circuits.

Different forms of synaptic plasticity underlie the formation of various types of memory. Long-term potentiation (LTP) in the hippocampus is considered the cellular mechanism of declarative memory (memory of facts and events). Strengthening of synaptic connections in the hippocampus enables encoding and storage of information about new episodes and knowledge. Emotional memory, particularly fear memory, is formed in the amygdala through LTP and LTD. Synaptic plasticity in the amygdala allows association of emotionally significant stimuli with specific responses and memories. Procedural memory, or memory of skills and habits, is linked to synaptic plasticity in the cerebellum and basal ganglia, where spike-timing-dependent plasticity (STDP) plays a vital role in motor learning and movement coordination. Short-term plasticity, such as synaptic facilitation and depression, may underlie sensory memory and working memory, providing temporary storage and processing of information over brief time intervals. Thus, the diversity of synaptic plasticity forms enables the brain to adapt to diverse learning demands and form a wide spectrum of memories—from transient sensory impressions to long-term knowledge and skills.

### Epigenetic mechanisms of long-term memory

Epigenetic modifications, such as DNA methylation and histone modifications, play a key role in stabilizing long-term synaptic changes and forming long-term memory. DNA methylation—the addition of a methyl group to cytosine—can alter DNA accessibility for transcription, affecting expression of genes critical for synaptic plasticity. Histone modifications—such as acetylation and methylation—regulate gene accessibility for transcription. These epigenetic changes, induced by synaptic activity, can persist over long periods, providing a molecular basis for long-term memory.

Synaptic plasticity, as a complex biological process, is under strict genetic control. Numerous genes encode proteins involved in various stages of synaptic plasticity—from synthesis and transport of receptors and neurotransmitters to signaling cascades and synaptic structural proteins. For example, genes encoding NMDA receptors (e.g., GRIN1, GRIN2A, GRIN2B) and AMPA receptors (e.g., GRIA1, GRIA2) play a crucial role in LTP and LTD. Genes encoding protein kinases and phosphatases such as CaMKII (CAMK2A, CAMK2B) and calcineurin (PPP3CA, PPP3CB) are also critically important for regulating synaptic strength. Genetic variations in these and other genes can influence individual differences in learning and memory capacity, as well as predisposition to neuropsychiatric disorders associated with synaptic plasticity impairments. Beyond genetic predisposition, epigenetic mechanisms such as DNA methylation and histone modifications (acetylation, methylation) play a vital role in long-term regulation of synaptic plasticity. These epigenetic changes can modulate expression of genes critical for synaptic function in response to neuronal activity and external stimuli, enabling long-term synaptic adaptation and memory consolidation.

### Role of the extracellular matrix

The extracellular matrix (ECM), a complex network of macromolecules surrounding cells in the brain, also plays a significant role in regulating synaptic plasticity. Perineuronal nets (PNNs), specialized ECM structures surrounding certain types of neurons, stabilize synaptic connections and restrict plasticity in the mature brain. Enzymes that degrade the ECM, such as matrix metalloproteinases (MMPs), can promote plasticity by breaking down PNNs and allowing synapses to change. Regulation of ECM structure and composition is an important mechanism controlling synaptic plasticity and formation of stable memories.

### Modern views on synaptic plasticity and extensions of Hebb’s rule

Modern research has significantly deepened our understanding of synaptic plasticity, revealing more complex and nuanced forms that extend beyond the classical Hebb’s rule. One such example is **spike-timing-dependent plasticity (STDP)**, where not only simultaneous activity but the precise temporal sequence of pre- and postsynaptic spikes determines the direction and magnitude of synaptic strength change. If a presynaptic spike precedes a postsynaptic spike within a specific time window (on the order of tens of milliseconds), the synapse is strengthened (analogous to LTP). Conversely, if the postsynaptic spike precedes the presynaptic spike, the synapse is weakened (analogous to LTD).

Mathematically, spike-timing-dependent plasticity (STDP) can be represented as:

$\Delta w_{ij} = F(\Delta t) = \begin{cases}
A_{+} \exp(-\Delta t / \tau_{+}), & \text{if } \Delta t > 0 \\
-A_{-} \exp(\Delta t / \tau_{-}), & \text{if } \Delta t < 0
\end{cases}$

where:

*   $\Delta w_{ij}$ — change in the strength of the synaptic connection between presynaptic neuron $j$ and postsynaptic neuron $i$.
*   $\Delta t = t_{post} - t_{pre}$ — time difference between the moment of spike generation by the postsynaptic neuron ($t_{post}$) and the moment of spike generation by the presynaptic neuron ($t_{pre}$).
*   $A_{+}$ — positive constant determining the maximum magnitude of synaptic potentiation.
*   $A_{-}$ — positive constant determining the maximum magnitude of synaptic depression.
*   $\tau_{+}$ — time constant for LTP (long-term potentiation).
*   $\tau_{-}$ — time constant for LTD (long-term depression).

This equation shows that the direction and magnitude of synaptic strength change depend on the temporal interval between spikes.

**Neuromodulators** such as **dopamine**, **serotonin**, **norepinephrine**, and **acetylcholine** play a crucial role in modulating synaptic plasticity and learning processes. These substances not only transmit signals between neurons but also alter the "rules of the game," influencing thresholds for LTP and LTD induction and the stability of synaptic changes. For example, dopamine released in response to unexpected rewards can enhance synaptic plasticity in involved neural circuits, facilitating reinforcement learning. Furthermore, research shows that synaptic plasticity is a dynamic process dependent on multiple factors, including **context**, **attention**, **stress levels**, and the organism’s **emotional state**. This underscores that learning is not a passive process but is actively modulated by internal states and external conditions.

## Learning in artificial neural networks and adaptation of Hebb’s rule

### Early models of learning in ANNs inspired by Hebb’s rule

Hebb’s rule served as a powerful inspiration for early learning models in artificial neural networks (ANNs). The core idea—that connections between neurons should be strengthened when they are co-active—was adapted into various learning algorithms.

*   **Hebbian network:** This is the simplest implementation of Hebb’s rule in ANNs. In a Hebbian network, weights between neurons are updated proportionally to the product of their activities. The mathematical formulation of the weight update rule is as follows:

    $\Delta w_{ij} = \eta \cdot y_i \cdot y_j$

    where $\Delta w_{ij}$ is the change in the weight between neurons $i$ and $j$, $\eta$ is a positive constant determining the learning rate, and $y_i$ and $y_j$ are the activities (outputs) of neurons $i$ and $j$, respectively. Thus, if both neurons are active simultaneously, the weight between them increases.

*   **Kohonen networks (Self-organizing maps):** Kohonen networks, also known as self-organizing maps (SOMs), are a type of unsupervised neural network. The learning principle in SOMs is based on competition and cooperation among neurons. When a data vector is presented to the network, the winning neuron (the neuron whose weights are most similar to the input vector) becomes activated. Then, according to principles inspired by Hebb’s rule, the weights of the winning neuron and its topological neighbors are adapted toward the input vector. This causes neurons responding to similar input data to become closer to each other in feature space, forming a map reflecting the structure of the input data.

    Weight updates for the winning neuron and its neighbors can be formalized as:

    $\mathbf{w}_{c}(t+1) = \mathbf{w}_{c}(t) + \alpha(t) \cdot h_{ci}(t) \cdot (\mathbf{x}(t) - \mathbf{w}_{c}(t))$

    where:
    *   $\mathbf{w}_{c}(t)$ — weight vector of the winning neuron $c$ at time $t$.
    *   $\mathbf{x}(t)$ — input vector at time $t$.
    *   $\alpha(t)$ — **learning rate** at time $t$.
    *   $h_{ci}(t)$ — **neighborhood function**, e.g., a Gaussian function: $h_{ci}(t) = \exp(-\frac{d(c, i)^2}{2\sigma(t)^2})$.

*   **Hopfield networks:** Hopfield networks are a type of recurrent neural network used as associative memory. Learning in Hopfield networks for storing specific patterns is performed using Hebb’s rule. Synaptic weights between neurons are set such that, when a partial or noisy pattern is presented, the network evolves into a state corresponding to one of the stored patterns. Hebb’s rule in this context ensures that neurons frequently activated together during presentation of a specific pattern develop strong connections, enabling the network to "recall" complete patterns.

    To store a set of patterns $\{\mathbf{\xi}^{(1)}, \mathbf{\xi}^{(2)}, ..., \mathbf{\xi}^{(p)}\}$, synaptic weights can be defined as:

    $w_{ij} = \frac{1}{N} \sum_{\mu=1}^{p} \xi_i^{(\mu)} \xi_j^{(\mu)}$  for $i \neq j$

    $w_{ii} = 0$

    where $\xi_i^{(\mu)}$ is the state of neuron $i$ in the $\mu$-th stored pattern.

### Modern ANN learning methods: gradient descent and backpropagation

In modern artificial neural networks (ANNs), both **supervised** and **unsupervised** learning are widely used. For training multilayer neural networks such as multilayer perceptrons, the dominant algorithm is **backpropagation**.

Backpropagation is an algorithm based on **gradient descent**. The goal of learning is to minimize the **loss function** $J(\mathbf{W})$, where $\mathbf{W}$ represents the set of all weights in the network. Weights are updated in the direction opposite to the gradient of the loss function:

$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \nabla J(\mathbf{W}^{(t)})$

where $\eta$ is the learning rate.

The gradient of the loss function with respect to a specific weight $w_{ij}^{(l)}$ in layer $l$ is computed using the chain rule, leading to the following weight update rule:

$\Delta w_{ij}^{(l)} = - \eta \cdot \delta_i^{(l)} \cdot a_j^{(l-1)}$

where $\delta_i^{(l)}$ is the local error for neuron $i$ in layer $l$, and $a_j^{(l-1)}$ is the activation of the neuron in the previous layer.

A key distinction from direct application of Hebb’s rule is that backpropagation-based learning is **error-driven**. Weight updates occur not merely based on simultaneous activity of connected neurons, but on how much the network’s prediction deviates from the desired output. Furthermore, supervised learning using backpropagation **requires labeled data**—a dataset of inputs with corresponding correct outputs.

### Connection between gradient descent and Hebbian learning principles

Despite differences, backpropagation learning can be viewed as a more sophisticated form of Hebbian learning. Weight updates in backpropagation also depend on the activity of connected neurons, albeit indirectly through the error gradient. Research aims to develop local learning rules in ANNs that are more biologically plausible and scalable while preserving the efficiency of modern algorithms. These efforts often seek to combine error-based learning principles with local Hebbian mechanisms to create more powerful and flexible artificial intelligence systems.

### Conclusion

Biological and artificial neural networks, despite sharing the common goal of information processing, exhibit both similarities and fundamental differences in their structure, function, and computational capabilities.

**Structural differences:**

*   Biological neurons possess a complex cellular structure, including soma, dendrites, axon, and synapses enabling transmission of electrochemical signals.
*   Artificial neurons are mathematical functions processing numerical values through inputs, weights, aggregation functions, and activation functions.
*   Signal transmission in biological neurons occurs via ions and neurotransmitters, whereas artificial neurons use mathematical operations over numbers.

**Functional differences:**

*   Biological neurons employ nonlinear signal transmission based on complex interactions of ion channels and synaptic plasticity.
*   Artificial neurons introduce nonlinearity via activation functions, enabling them to model complex dependencies.
*   Biological neurons learn through synaptic plasticity—including long-term potentiation (LTP) and depression (LTD)—as well as **spike-timing-dependent plasticity (STDP)**, while artificial neural networks learn via optimization algorithms such as gradient descent and backpropagation.
*   Biological neural networks operate continuously and asynchronously, whereas artificial neural networks often process data discretely.
*   In biological neurons, **modulators** such as dopamine, serotonin, and norepinephrine influence plasticity.

**Similarities:**

*   Both types of neurons perform the functions of receiving, processing, and transmitting information.
*   Both biological and artificial neurons integrate input signals, although summation mechanisms differ.
*   Both types of neurons exhibit nonlinearity, enabling them to model complex dependencies.
*   Both systems are capable of learning and adaptation by adjusting parameters (synaptic strength or weights) in response to experience.
*   Both types of neural networks use **Hebb’s principle** (or its extensions) for learning, where simultaneous activation of neurons leads to strengthening of their connection.

**Mathematical models:**

*   The dynamics of the membrane potential in biological neurons are described by the **Hodgkin-Huxley model**, which accounts for sodium, potassium, and leak ion currents.
    *   $$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$
*   In contrast, the operation of an artificial neuron is described by a simple mathematical model:
    *   $$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$
    where $y$ is the neuron output, $\phi$ is the activation function, $w_i$ is the weight of the $i$-th input, $x_i$ is the value of the $i$-th input, and $b$ is the bias.

**Advantages and limitations:**

*   Biological neurons exhibit high adaptability, energy efficiency, and parallel information processing, but have relatively slow signal transmission speeds.
*   Artificial neurons provide high computational speed, ease of scaling, and reproducibility, but consume significant energy and have limited generalization capability.

**Key takeaways:**

*   An artificial neuron is a simplified model of a biological neuron that nevertheless successfully mimics core principles of information processing.
*   It is essential to recognize the fundamental differences and limitations of artificial models and strive to create more biologically realistic neural networks.
*   Future research focuses on developing neuromorphic chips that emulate the architecture and dynamics of biological neural networks.

In conclusion, **biological and artificial neural networks use similar principles of information processing, but biological networks possess far more complex structure, dynamics, and adaptive capacity, while artificial networks are more efficient in terms of speed and scalability. Understanding these similarities and differences is essential for advancing both neuroscience and artificial intelligence**.


### References

[0] Eremin, A. L. Noogenesis and the Theory of Intelligence / A. L. Eremin. — 4th ed. — Krasnodar: SovKub, 2005. — 356 pp.

[1] Nature Neuroscience: The Hodgkin-Huxley theory of the action potential.

[2] Keener, J., & Sneyd, J. (2008). Mathematical Physiology: I: Cellular Physiology. Springer.

[3] Komarцova, L. G. Neurocomputers / L. G. Komarцova, A. V. Maksimov. — 1st ed. — Moscow: Bauman MSTU, 2004. — 400 pp.

[4] Cybenko, G. Approximation by Superpositions of a Sigmoidal Function / G. Cybenko. — Text: direct // Mathematics of Control, Signals, and Systems. — 1989. — № 2. — P. 303–314.