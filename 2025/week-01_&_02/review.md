# Comparative Analysis of Biological and Artificial Neural Networks

# **The Biological Neuron and Its Components**

## Introduction
Neural networks are fundamental structures underlying cognitive processes in biological organisms and have inspired scientists to develop mathematical models known as artificial neural networks. Comparing these two approaches provides insight into the nature of thought, learning systems, and the possibility of replicating intelligent behavior. This text examines the basic principles of biological neural network organization and artificial system architectures, highlighting their similarities and differences in terms of structure, dynamics, learning, and computational capabilities.

### Structure of the Biological Neuron

Biological neural networks form the foundational basis of the nervous system in living organisms, enabling information processing, transmission, and storage. Their structural and functional unit is the neuron—a highly specialized cell designed to generate and propagate electrochemical signals. Understanding neuronal structure is critical for grasping the operating principles of biological neural networks and their modeling in artificial systems.

1. **Soma (cell body)**: Contains the nucleus and major organelles, supporting metabolic and regulatory processes.  
2. **Dendrites**: Branching structures through which the neuron receives input signals from other cells. Synaptic contacts primarily form on dendrites.  
3. **Axon**: A single large process along which the action potential (spike) propagates over long distances to synapses.  
4. **Synapses**: Sites of chemical or electrical signal transmission. Synaptic vesicles release neurotransmitters (e.g., glutamate, GABA, acetylcholine), which bind to receptors on the postsynaptic membrane.

![Structure_of_biological_neuron](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-01_%26_02/assets/Page_1.png)

**General Mathematical Neuron Model:**

The fundamental and widely used mathematical model describing action potential dynamics in living neurons—including human neurons—is the **Hodgkin-Huxley model**. Developed based on experiments with the squid giant axon, this model describes how the neuronal membrane potential changes over time, accounting for sodium ($Na^+$) and potassium ($K^+$) ion flows through voltage-gated ion channels, as well as leak current. [1](https://www.nature.com/articles/nn1100_1165)

The core equation of the model expresses the balance of transmembrane currents:

$$C_m \frac{dV_m}{dt} = -I_{Na} - I_K - I_L + I_{ext}$$

where:  
- $C_m$ — specific membrane capacitance (F/m²).  
- $\frac{dV_m}{dt}$ — rate of change of membrane potential $V_m$ (V).  
- $I_{Na}$, $I_K$ — sodium and potassium ionic currents, respectively (A/m²).  
- $I_L$ — leak current (A/m²).  
- $I_{ext}$ — externally applied current (A/m²).

Ionic currents are described by the following equations:

$$I_{Na} = g_{Na} m^3 h (V_m - E_{Na})$$

$$I_K = g_K n^4 (V_m - E_K)$$

$$I_L = g_L (V_m - E_L)$$

where:  
- $g_{Na}$, $g_K$, $g_L$ — maximum conductances for sodium, potassium, and leak currents, respectively (S/m²). These parameters reflect the density of corresponding ion channels in the membrane.  
- $E_{Na}$, $E_K$, $E_L$ — equilibrium (Nernst) potentials for sodium, potassium, and leak currents, respectively (V). These are determined by intra- and extracellular ion concentrations.  
- $m$ — sodium channel activation variable (dimensionless, 0–1), representing the probability of sodium channel opening.  
- $h$ — sodium channel inactivation variable (dimensionless, 0–1), representing the probability that sodium channels are not blocked.  
- $n$ — potassium channel activation variable (dimensionless, 0–1), representing the probability of potassium channel opening.

The activation and inactivation variables $m$, $h$, and $n$ are functions of membrane potential and time, governed by first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_m) (1-m) - \beta_m(V_m) m$$

$$\frac{dh}{dt} = \alpha_h(V_m) (1-h) - \beta_h(V_m) h$$

$$\frac{dn}{dt} = \alpha_n(V_m) (1-n) - \beta_n(V_m) n$$

where $\alpha(V_m)$ and $\beta(V_m)$ are voltage-dependent opening and closing rates of ion channels, determined experimentally.

**Biological Description of the Hodgkin-Huxley Model:**

At rest, the neuron maintains specific ion gradients (high intracellular potassium and extracellular sodium), resulting in a negative resting membrane potential (approximately –70 mV) inside the cell relative to the outside. When the membrane potential becomes less negative and reaches a threshold—typically around –55 mV—it triggers an avalanche-like opening of voltage-gated sodium channels, causing positive sodium ions to rush into the soma (more precisely, into the axon). This ion movement is driven by the electrochemical gradient, which includes both concentration differences and electrical charge. This leads to rapid depolarization, making the intracellular axonal potential positive. Shortly thereafter, sodium channels inactivate, halting further sodium influx. At this point, voltage-gated potassium channels open more slowly, allowing positive potassium ions to exit the cell. This efflux of positive charge causes repolarization, returning the membrane potential to negative values. Brief hyperpolarization may sometimes occur because certain voltage-gated potassium channels remain open longer than needed to restore the resting potential, temporarily making the membrane potential more negative than usual. Ultimately, the sodium-potassium pump (which actively extrudes sodium and imports potassium) helps restore and maintain ion gradients, returning the axon to a stable negative resting potential. These processes occur primarily in the axon hillock and axon, where the action potential is generated and propagated.

**Neuronal Signal Transmission:**

*Intraneuronal signal transmission (over long distances):*  
- The action potential is an active signaling mechanism. Once initiated at the axon initial segment (axon hillock), it regenerates itself along the entire axon length to the nerve terminals. This occurs through the sequential opening and closing of voltage-gated ion channels along the axon. Each axonal "point" generates a new action potential, effectively "pushing" the signal forward.

*Interneuronal signal transmission (synaptic transmission):*  
- When the action potential reaches the axon terminals (presynaptic terminals), it triggers synaptic transmission.  
- Depolarization of the presynaptic terminal (caused by the arriving action potential) opens voltage-gated calcium channels.  
- Calcium ions ($Ca^{2+}$) enter the presynaptic terminal.  
- Calcium influx causes neurotransmitter-containing vesicles to fuse with the presynaptic membrane.  
- Neurotransmitters are released into the synaptic cleft (the space between two neurons).  
- Neurotransmitters bind to receptors on the postsynaptic neuron’s membrane.  
- Neurotransmitter-receptor binding can alter the postsynaptic membrane’s ion permeability, leading to depolarization (excitatory postsynaptic potential, EPSP) or hyperpolarization (inhibitory postsynaptic potential, IPSP).  
- If the summed depolarization in the postsynaptic neuron reaches threshold, it too will generate an action potential, propagating the signal further.

Thus, the action potential serves two key signaling functions:  
- It ensures reliable long-distance signal transmission within a single neuron (without it, signals would simply decay).  
- It initiates synaptic transmission, enabling neurons to "communicate" with each other. The action potential acts as the "trigger" for neurotransmitter release.

**Relevance to Human Neurons:**

Although the original model was developed for the squid axon, its principles underlie our understanding of action potential generation in most excitable cells, including human neurons. Model parameters ($g_{Na}$, $g_K$, $g_L$, kinetics of $m$, $h$, $n$) may vary depending on neuron type and functional characteristics. Current research actively adapts and parameterizes the Hodgkin-Huxley model for various human neuron types using experimental data from patch-clamp techniques.

**Significance and Limitations:**

The Hodgkin-Huxley model is foundational because it:  
- **Is grounded in biophysical processes**: It describes ion channel behavior, which is central to action potential generation.  
- **Quantitatively reproduces action potential shape**: It accurately models membrane potential dynamics.  
- **Serves as a basis for more complex models**: Many modern neuron models build upon Hodgkin-Huxley principles.

However, its limitations must be acknowledged:  
- **Describes a point neuron**: It ignores the neuron’s spatial structure (dendrites, axon) and signal propagation along it. Multi-compartment models are used to simulate these aspects.  
- **Simplifies ion channel behavior**: It treats ion channels as homogeneous populations, neglecting subunit structure and stochastic opening/closing.  
- **Does not include all ion channel types**: Real neurons express many ion channel types that influence excitability and firing patterns.  
- **Parameters can be difficult to determine**: Precise parameterization for a specific human neuron requires complex experimental studies.

**Modern Extensions:**

Numerous extensions of the Hodgkin-Huxley model now exist, aiming to overcome its limitations and better capture neuronal behavior:  
- **Multi-compartment models**: Divide the neuron into compartments (soma, dendrites, axon) and simulate electrical signal propagation between them.  
- **Models incorporating diverse ion channel types**: Add equations for other ion channels (e.g., calcium, chloride, various potassium subtypes).  
- **Models integrating synaptic transmission**: Include synaptic mechanisms to simulate neuron-neuron interactions.  
- **Models incorporating intracellular processes**: Include biochemical processes affecting ion channel function and neuronal excitability.

In conclusion, despite its age, the Hodgkin-Huxley model remains a cornerstone of neuronal mathematical modeling, including for human neurons. Understanding its principles and limitations is essential for interpreting simulation results and developing more sophisticated, realistic models.

## Soma (Cell Body): Center of Metabolism and Signal Integration

The soma, or cell body, is the central part of the neuron, containing the nucleus and essential organelles required to sustain cellular life.

**Internal Neuron Structure:**  
- **Nucleus**: Contains the neuron’s genetic material (DNA), organized into chromosomes. The nucleus controls protein synthesis necessary for neuronal structure and function—including enzymes, structural proteins, and receptors. DNA transcription into mRNA and subsequent ribosomal translation ensure continuous renewal and maintenance of cellular components. Nuclear gene expression regulation plays a key role in neuronal adaptation to changing conditions and in learning and memory processes.  
- **Organelles**: The soma contains a typical set of eukaryotic organelles, each performing specialized functions:  
  - **Mitochondria**: Produce energy in the form of ATP via cellular respiration. Neurons exhibit high metabolic activity and require substantial energy to maintain ion gradients and support signaling processes.  
  - **Endoplasmic reticulum (ER)**: Participates in protein and lipid synthesis and transport. Rough ER, studded with ribosomes, synthesizes proteins destined for membranes and secretion. Smooth ER is involved in lipid synthesis and detoxification.  
  - **Golgi apparatus**: Modifies, sorts, and packages proteins and lipids for transport to other cellular regions or for secretion. It contributes to lysosome and synaptic vesicle formation.  
  - **Lysosomes**: Contain hydrolytic enzymes needed to break down cellular waste and damaged organelles (autophagy).  
  - **Cytoskeleton**: Composed of a network of protein filaments (microtubules, actin filaments, neurofilaments) that provide structural support, maintain neuronal shape, and facilitate intracellular transport (axonal transport).  
- **Cytoplasm**: A gel-like substance filling the soma, housing organelles and containing dissolved molecules such as ions, enzymes, and nutrients.

The soma also plays a crucial role in integrating incoming signals. Potentials generated in dendrites propagate toward the soma, where they are summed. If the total potential reaches threshold at the axon hillock, an action potential is generated.

**Soma (Cell Body) – Mathematical Model:**

To accurately and realistically describe the soma’s integrative function, one must account for the fact that the neuronal membrane is not a passive element. **Active membrane properties**, arising from diverse voltage-gated ion channels, are critical for processing incoming signals. Within the **single-compartment approach**—which assumes uniform potential distribution inside the soma—the mathematical formalization of somatic membrane potential dynamics is based on the following principles:

$$C_m \frac{dV_{soma}}{dt} = -I_{ion} + I_{syn}$$

where:  
- $C_m$ — soma membrane capacitance (F).  
- $\frac{dV_{soma}}{dt}$ — rate of change of somatic membrane potential (V/s).  
- $I_{ion}$ — total ionic current across the soma membrane (A). This current includes flows of various ions—such as sodium, potassium, calcium, and others—depending on neuron type.  
- $I_{syn}$ — total synaptic current entering the soma from dendrites (A).

**Ionic Current ($I_{ion}$) Detailing:**

The ionic current is the sum of individual ion flows through various ion channel types present in the soma membrane. As an example, consider the main currents found in most neurons:

$$I_{ion} = I_{Na} + I_K + I_L + I_{Ca} + ...$$

where:  
- $I_{Na}$ — sodium current through voltage-gated sodium channels.  
- $I_K$ — potassium current through various voltage-gated potassium channels (e.g., delayed rectifier, A-type, etc.).  
- $I_L$ — leak current, representing passive ion flow across the membrane.  
- $I_{Ca}$ — calcium current through voltage-gated calcium channels (various types, such as L-type, N-type, T-type, P/Q-type).  
- `...` — indicates the possible presence of other ionic currents, such as chloride currents, hyperpolarization-activated currents (Ih), etc., depending on neuron type.

Each ionic current is described by an equation of the form:

$$I_{ion\_type} = g_{ion\_type} \cdot m^p \cdot h^q \cdot (V_{soma} - E_{ion\_type})$$

where:  
- $g_{ion\_type}$ — maximum conductance for this ion channel type (S), reflecting channel density in the membrane.  
- $m$ — channel activation variable (dimensionless, 0–1), representing the probability of channel opening.  
- $h$ — channel inactivation variable (dimensionless, 0–1), representing the probability the channel is not blocked. Not all channel types have an inactivation variable.  
- $p$ and $q$ — stoichiometric coefficients indicating the number of activation and inactivation molecules required for channel opening.  
- $V_{soma}$ — somatic membrane potential (V).  
- $E_{ion\_type}$ — equilibrium (Nernst) potential for this ion (V).

Activation ($m$) and inactivation ($h$) variables are functions of membrane potential and time, governed by first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_{soma}) (1-m) - \beta_m(V_{soma}) m$$
$$\frac{dh}{dt} = \alpha_h(V_{soma}) (1-h) - \beta_h(V_{soma}) h$$

where $\alpha(V_{soma})$ and $\beta(V_{soma})$ are voltage-dependent channel opening and closing rates, determined experimentally and often represented by various functional forms.

**Synaptic Current ($I_{syn}$) Detailing:**

The synaptic current is the sum of currents from all synapses located on the soma (and, in more general models, from dendrites influencing somatic potential):

$$I_{syn} = \sum_{i} I_{syn\_i}$$

Each synaptic current can be modeled as:

$$I_{syn\_i} = g_{syn\_i}(t) \cdot (V_{soma} - E_{syn\_i})$$

where:  
- $g_{syn\_i}(t)$ — conductance of the $i$-th synapse, time-dependent and determined by neurotransmitter release and binding to postsynaptic receptors. This conductance is typically modeled as a function with rapid rise and slow decay.  
- $V_{soma}$ — somatic membrane potential (V).  
- $E_{syn\_i}$ — equilibrium potential for this synapse type (V). For excitatory synapses (e.g., glutamatergic), $E_{syn}$ is usually near 0 mV; for inhibitory synapses (e.g., GABAergic), $E_{syn}$ is close to the chloride or resting potential.

Synaptic conductance $g_{syn\_i}(t)$ is often modeled using kinetic schemes describing neurotransmitter-receptor binding and ion channel opening, or via alpha functions and their variants.

**Action Potential Generation Threshold:**

In single-compartment models, action potential generation is usually not modeled in detail within the soma. Instead, it is assumed that if the somatic membrane potential $V_{soma}(t)$ reaches a threshold value $\theta_{AP}$, an action potential is initiated at the axon hillock. However, more sophisticated models may include detailed modeling of axon hillock ion channels to accurately reproduce spike initiation.

**Advantages and Limitations of This Model:**

- **Advantages**:  
  - Accounts for active soma membrane properties, enabling simulation of local potential generation and more realistic signal integration.  
  - Can reproduce diverse neuronal firing patterns depending on ion channel composition and parameters.  
- **Limitations**:  
  - Assumes the soma is isopotential, which may not hold for neurons with large or complex somata.  
  - Model parameters (maximum conductances, ion channel kinetics) must be experimentally determined for specific neuron types, which can be challenging.

**More Complex Models:**

For more accurate simulation of somatic electrical activity—especially when the isopotential assumption fails—**multi-compartment models** are used. In these models, the soma is divided into several interconnected compartments, each described by a set of equations similar to those above, incorporating passive membrane properties linking the compartments.

**Conclusion:**

The presented single-compartment mathematical model of the soma, incorporating active membrane properties, is far more relevant and comprehensive than a simple postsynaptic potential (PSP) summation model. It enables simulation of somatic membrane potential dynamics driven by various ion channels and synaptic inputs. For scientific work requiring detailed understanding of somatic electrical activity, using such a model—or its multi-compartment extensions—is essential. Importantly, model accuracy directly depends on correct parameterization based on experimental data for the specific neuron type.


## Dendrites: Receiving and Integrating Incoming Signals

Dendrites are numerous, typically short and highly branched processes extending from the soma. Their primary function is to receive incoming signals from other neurons.

**Internal Dendritic Structure:**  
- **Dendritic morphology**: The shape and size of dendritic trees vary significantly depending on neuron type and function. Their complex branching structure increases surface area for receiving synaptic inputs.  
- **Dendritic spines**: Many dendrites are covered with small protrusions called dendritic spines. Spines are the primary sites of excitatory synapse formation. Their dynamic structure (changes in shape and number) plays a crucial role in synaptic plasticity, which underlies learning and memory. A spine consists of a head, neck, and base, and its morphology influences the electrical properties of the synaptic signal.  
- **Synaptic contacts**: Most synaptic contacts form on dendrites—either on smooth surfaces or on spines. The presynaptic neuron releases neurotransmitters that bind to receptors on the dendritic postsynaptic membrane, altering its ion permeability.  
- **Local potentials**: Neurotransmitter binding to dendritic receptors generates local changes in membrane potential—postsynaptic potentials (PSPs). Excitatory PSPs (EPSPs) depolarize the membrane, bringing it closer to the action potential threshold. Inhibitory PSPs (IPSPs) hyperpolarize the membrane, moving it further from threshold.  
- **Signal integration**: Dendrites not only receive signals but also participate in their integration. Spatial summation occurs when signals from multiple synapses at different dendritic locations reach the soma nearly simultaneously. Temporal summation occurs when successive signals from a single synapse reach the soma with minimal delay. Dendritic shape and electrical properties (e.g., presence of voltage-gated ion channels) affect integration efficiency.

**Dendrites – Mathematical Model:**

**Fundamental Membrane Potential Equation** [2]

The dynamics of dendritic membrane potential at any point can be described using charge conservation and modeling the membrane as an electrical circuit:

$$C_m \frac{dV_m(x,t)}{dt} = -I_{ion}(x,t) + I_{ext}(x,t)$$

where:  
- $V_m(x,t)$ — membrane potential at location $x$ and time $t$.  
- $C_m$ — membrane capacitance per unit area.  
- $I_{ion}(x,t)$ — total ionic current across the membrane at location $x$ and time $t$.  
- $I_{ext}(x,t)$ — external current applied to the membrane at location $x$ and time $t$ (e.g., synaptic current).

The ionic current $I_{ion}$ includes leak and various ion channel currents:

$$I_{ion}(x,t) = g_L (V_m(x,t) - E_L) + \sum_i I_i(x,t)$$

where:  
- $g_L$ — specific leak conductance.  
- $E_L$ — reversal potential for leak current.  
- $I_i(x,t)$ — current through the $i$-th type of ion channel, which may depend on voltage and ion concentration.

**Modeling Synaptic Current**

Synaptic current $I_{syn}(t)$, a special case of $I_{ext}(x,t)$, arises when neurotransmitters bind to postsynaptic receptors. It can be modeled in various ways reflecting binding kinetics and ion channel conductance:

- **Instantaneous conductance jump**:

    $$I_{syn}(t) = g_{syn} \cdot s(t) \cdot (V_m(t) - E_{syn})$$

    where:  
    - $g_{syn}$ — maximum synaptic conductance.  
    - $s(t)$ — binary function equal to 1 during synaptic activity and 0 otherwise.  
    - $E_{syn}$ — synaptic current reversal potential.

- **First-order kinetics model**:

    $$I_{syn}(t) = g_{syn} \cdot r(t) \cdot (V_m(t) - E_{syn})$$

    where $r(t)$ describes the fraction of open ion channels and obeys the differential equation:

    $$\frac{dr}{dt} = \alpha [T](t) (1-r) - \beta r$$

    where:  
    - $\alpha$ — neurotransmitter binding rate.  
    - $\beta$ — neurotransmitter unbinding rate.  
    - $[T](t)$ — neurotransmitter concentration in the synaptic cleft.

**Derivation of Postsynaptic Potential (PSP) Models**

The PSP models introduced earlier are approximations of membrane potential dynamics in response to synaptic current. They can be derived from more fundamental equations under specific assumptions.

- **Exponential PSP model**: This model arises when assuming synaptic conductance rises and decays exponentially. Considering a simplified leaky membrane model with synaptic current yields a solution of the form:

    $$PSP(t) \propto e^{-t/\tau_{m}} \int_0^t I_{syn}(\tau) e^{\tau/\tau_{m}} d\tau$$

    where $\tau_m = C_m / g_L$ is the membrane time constant. Approximating the synaptic current shape leads to expressions similar to the exponential PSP model presented earlier.

- **Alpha-function PSP**: This model is often used for AMPA receptor-mediated PSPs. It can be derived from the first-order kinetics model under certain assumptions about $[T](t)$.

**Important note**: These models are phenomenological and do not always accurately reflect the complex kinetics of ion channels and receptors.

**Spatial Signal Integration and Cable Theory**

Dendrites are not isopotential structures. Incoming synaptic signals propagate along dendritic branches, attenuating with distance from their origin. **Cable theory** describes this process.

A dendritic branch is modeled as a cylindrical cable with specific membrane resistance $R_m$, specific axial resistance $R_a$, and specific membrane capacitance $C_m$. Changes in membrane potential $v(x,t)$ along the dendrite are described by the **cable equation**:

$$\lambda^2 \frac{\partial^2 v(x,t)}{\partial x^2} - \tau_m \frac{\partial v(x,t)}{\partial t} - v(x,t) = r_m i_{ext}(x,t)$$

where:  
- $v(x,t) = V_m(x,t) - V_{rest}$ — deviation of membrane potential from resting potential.  
- $\lambda = \sqrt{\frac{r_m}{r_a}}$ — **space constant** (length constant), determining how far a signal propagates along the dendrite. Here $r_m = R_m / \pi d$, $r_a = R_a / (\pi d^2 / 4)$, where $d$ is dendrite diameter.  
- $\tau_m = r_m c_m$ — **membrane time constant**. Here $c_m = C_m \pi d$.  
- $r_m$ — membrane resistance per unit length.  
- $i_{ext}(x,t)$ — external current per unit length (e.g., synaptic current).

Solving this equation allows calculation of each synaptic input’s contribution to the potential at any dendritic location, including the soma. Spatial summation results from linear superposition of these contributions (in passive dendrites).

**Influence of Dendritic Morphology**

The complex branching structure of dendritic trees significantly affects signal integration. Cable theory can be extended to analyze branched structures by treating each branch as a separate cable with appropriate boundary conditions at branch points.

- **Input resistance**: Dendritic morphology determines input resistance, which affects PSP amplitude. More branched dendrites with smaller branch diameters typically have higher input resistance.  
- **Electrotonic distance**: The effective distance between synapses and the soma is determined not only by physical distance but also by electrotonic distance, which accounts for signal attenuation. Synapses on distal branches exert less influence on somatic potential due to greater electrotonic distance.

**Temporal Signal Integration**

Temporal summation occurs when successive PSPs from one or more synapses reach the soma with minimal delay. Mathematically, this can be described as superposition of PSPs, accounting for their temporal dynamics:

$$V_{soma}(t) = \sum_i w_i \cdot PSP_i(t - t_i)$$

where:  
- $w_i$ — effective weight of the $i$-th synapse, accounting for its strength and electrotonic distance to the soma.  
- $PSP_i(t - t_i)$ — PSP waveform evoked by the $i$-th synapse, accounting for its activation time $t_i$.

More accurate modeling requires considering interactions between successive PSPs, especially if they occur rapidly enough to overlap.

**Active Dendrites: Accounting for Nonlinearities**

Unlike passive cables, dendrites in many neuron types contain **voltage-gated ion channels**. These channels introduce nonlinearity into signal integration, enabling dendrites to generate local spikes or amplify incoming signals.

Modeling active dendrites requires incorporating Hodgkin-Huxley equations or their modifications to describe various ion channel dynamics (e.g., sodium, potassium, calcium):

$$C_m \frac{dV_m}{dt} = -g_L(V_m - E_L) - g_{Na} m^3 h (V_m - E_{Na}) - g_K n^4 (V_m - E_K) - ... + I_{syn}$$

where $m$, $h$, $n$ are ion channel activation/inactivation variables described by first-order differential equations.

Active channels enable dendrites to perform more complex computations, such as threshold logic at the level of individual branches.

**Compartmental Modeling**

**Compartmental modeling** is commonly used to simulate complex dendritic trees with nonlinear properties. The dendritic tree is divided into many small isopotential segments (compartments). Each compartment is described by a system of differential equations accounting for transmembrane currents and inter-compartmental currents.

This approach allows numerical solution of equations for complex geometries and inclusion of various ion channel types in different dendritic regions.

**Influence of Dendritic Spines**

Dendritic spines, the sites of most excitatory synapses, play a key role in synaptic transmission and plasticity. Their morphology (head and neck size) affects:

- **Input resistance**: A spine with a narrow neck increases local input resistance, resulting in greater depolarization for a given synaptic current.  
- **Electrical isolation**: The spine neck can electrically isolate the head from the main dendritic branch, allowing spines to function as relatively independent computational units.  
- **Calcium diffusion**: Spine morphology influences calcium ion diffusion, which plays a key role in synaptic plasticity.

Mathematical spine modeling often treats them as additional compartments with specific geometric and electrical properties.

**Stochasticity of Synaptic Transmission**

It is important to note that synaptic transmission is a stochastic process. Neurotransmitter release and ion channel opening occur with certain probabilities. More realistic modeling can employ probabilistic synaptic transmission models—for example, Poisson models for neurotransmitter release.

### Conclusion

The presented mathematical formalization demonstrates the complexity and multifaceted nature of dendritic processes. From fundamental membrane potential equations to active dendrite models and compartmental modeling, a wide range of tools exists for studying dendritic signal integration.

**Limitations and Future Directions:**  
- **Simplifications**: Most models simplify biological reality. For example, exact dendritic geometry and ion channel distribution are often unknown.  
- **Parameterization**: Determining precise model parameter values (channel conductances, kinetic constants) remains challenging.  
- **Computational complexity**: Simulating complex dendritic trees with active properties requires significant computational resources.

## Axon: Long-Distance Action Potential Transmission

The axon is typically a single, long, thin process extending from the soma at the axon hillock. Its primary function is to transmit electrical signals (action potentials or spikes) over long distances to other neurons, muscle cells, or glands.

**Internal Axonal Structure:**  
- **Axon hillock**: A specialized region of the soma where incoming signals are integrated and the action potential is initiated. The axon hillock contains a high concentration of voltage-gated sodium channels, making it the most excitable part of the neuron.  
- **Axon initial segment**: Directly adjacent to the axon hillock, this is where the action potential is usually generated.  
- **Myelin sheath**: In many neurons, the axon is covered by a myelin sheath formed by glial cells (oligodendrocytes in the central nervous system and Schwann cells in the peripheral nervous system). Myelin is a lipid-based insulator that greatly increases action potential conduction velocity.  
- **Nodes of Ranvier**: Periodic gaps in the myelin sheath where the axon remains uninsulated. Voltage-gated sodium channels are concentrated at the nodes, enabling the action potential to "jump" from one node to the next (saltatory conduction), significantly increasing transmission speed.  
- **Axon terminals (presynaptic endings)**: At the distal end, the axon branches into multiple terminals. Each terminal forms a synapse with another cell.  
- **Synaptic vesicles**: Membrane-bound structures within axon terminals containing neurotransmitters.  
- **Action potential transmission mechanism**: When the action potential reaches the axon terminal, membrane depolarization opens voltage-gated calcium channels. Calcium influx triggers synaptic vesicle fusion with the presynaptic membrane (exocytosis) and neurotransmitter release into the synaptic cleft.

**Axon – Mathematical Model:**

**Action Potential Generation: Hodgkin-Huxley Model**

The foundational model describing action potential generation is the **Hodgkin-Huxley (HH) model**. Based on experimental data from the squid giant axon, it describes membrane potential $V_m$ dynamics using a system of nonlinear differential equations:

$$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$

where $I_{ion}$ is the total ionic current, including sodium ($I_{Na}$), potassium ($I_K$), and leak ($I_L$) currents:

$$I_{ion} = I_{Na} + I_K + I_L$$

Each ionic current is modeled as:

$$I_{Na} = g_{Na} m^3 h (V_m - E_{Na})$$
$$I_K = g_K n^4 (V_m - E_K)$$
$$I_L = g_L (V_m - E_L)$$

where:  
- $g_{Na}$, $g_K$, $g_L$ — maximum conductances for sodium, potassium, and leak channels, respectively.  
- $E_{Na}$, $E_K$, $E_L$ — equilibrium potentials for sodium, potassium, and leak ions, respectively.  
- $m$, $h$, $n$ — dimensionless variables describing sodium channel activation, sodium channel inactivation, and potassium channel activation, respectively. They obey first-order differential equations:

$$\frac{dm}{dt} = \alpha_m(V_m) (1-m) - \beta_m(V_m) m$$
$$\frac{dh}{dt} = \alpha_h(V_m) (1-h) - \beta_h(V_m) h$$
$$\frac{dn}{dt} = \alpha_n(V_m) (1-n) - \beta_n(V_m) n$$

where $\alpha(V_m)$ and $\beta(V_m)$ are voltage-dependent channel opening/closing rates determined experimentally.

**Simplifications and Alternative Action Potential Models**

Despite its accuracy, the Hodgkin-Huxley model is computationally expensive. Simplified models are used for specific tasks:

- **FitzHugh-Nagumo model**: A two-dimensional model describing membrane potential dynamics and a recovery variable mimicking sodium channel inactivation and potassium channel activation.

    $$\frac{dV}{dt} = c(V - \frac{V^3}{3} + R + I_{ext})$$
    $$\frac{dR}{dt} = - \frac{1}{c} (V - a + bR)$$

    where $a$, $b$, $c$ are model parameters.

- **Integrate-and-Fire models**: Abstract models where membrane potential integrates input current and generates a spike upon reaching threshold, after which the potential resets. Variants include models with adaptive thresholds or subthreshold oscillations.

    $$ \tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R I_{ext}$$
    If $V_m(t) \geq V_{thresh}$, then $V_m \rightarrow V_{reset}$.

Model choice depends on required accuracy and computational resources.

**Action Potential Propagation in Unmyelinated Axons**

Action potential propagation along the axon can be described using **cable theory**, similar to dendrites, but accounting for voltage-gated ion channels. The equation for membrane potential $v(x,t)$ along the axon is:

$$c_m \frac{\partial v}{\partial t} = \frac{1}{r_a} \frac{\partial^2 v}{\partial x^2} - I_{ion}(v)$$

where $I_{ion}(v)$ is the nonlinear ionic current described by Hodgkin-Huxley equations (or another ion channel model). This is a nonlinear parabolic partial differential equation, and analytical solutions are difficult. Numerical methods are typically used.

**Saltatory Conduction in Myelinated Axons**

The myelin sheath greatly increases conduction velocity through **saltatory conduction**. Myelin acts as an insulator, increasing membrane resistance and decreasing membrane capacitance in internodes. Action potentials are generated only at **Nodes of Ranvier**, where voltage-gated sodium channels are concentrated.

**Mathematical Model of Saltatory Conduction**

Modeling saltatory conduction involves considering both myelinated internodes and unmyelinated Nodes of Ranvier.

- **Internodes**: Passive electrical properties dominate in internodes. Potential propagation is described by the cable equation without active ionic currents, but with high $r_m$ and low $c_m$ values.

- **Nodes of Ranvier**: Membrane potential dynamics at nodes are described by the Hodgkin-Huxley model (or equivalent), as action potentials are generated here.

A myelinated axon model can be represented as a series of segments representing internodes and nodes. Current propagates passively through internodes and is actively amplified at nodes.

Mathematically, this is described by a system of coupled differential equations for each segment. For the $i$-th Node of Ranvier:

$$C_{m,node} \frac{dV_{m,i}}{dt} = -I_{ion,node}(V_{m,i}) + I_{axial,i-1 \rightarrow i} - I_{axial,i \rightarrow i+1}$$

where $I_{axial}$ is the axial current between segments, depending on potential differences and axial resistance.

For the $j$-th internode:

$$C_{m,internode} \frac{dV_{m,j}}{dt} = \frac{V_{m,j-1} - V_{m,j}}{R_{axial,j-1 \rightarrow j}} - \frac{V_{m,j} - V_{m,j+1}}{R_{axial,j \rightarrow j+1}}$$

where $R_{axial}$ is the internodal axial resistance.

**Signal Transmission at Axon Terminals: Synaptic Transmission**

When the action potential reaches the axon terminal, it triggers **synaptic transmission**. Presynaptic membrane depolarization opens **voltage-gated calcium channels**.

Intraterminal calcium ion concentration $[Ca^{2+}]_i$ dynamics can be described by:

$$\frac{d[Ca^{2+}]_i}{dt} = - \beta I_{Ca} - \frac{[Ca^{2+}]_i - [Ca^{2+}]_{rest}}{\tau_{Ca}}$$

where:  
- $\beta$ — constant linking calcium current to concentration change.  
- $I_{Ca}$ — calcium channel current, modeled similarly to sodium/potassium channels in the HH model.  
- $[Ca^{2+}]_{rest}$ — resting calcium concentration.  
- $\tau_{Ca}$ — calcium buffering and removal time constant.

Calcium influx triggers synaptic vesicle fusion with the presynaptic membrane and neurotransmitter release into the synaptic cleft.

**Modeling Neurotransmitter Release**

Neurotransmitter release is a probabilistic process. Release probability $P_{release}$ depends on presynaptic terminal calcium concentration. Various models can be used:

- **Threshold model**: Release occurs when calcium concentration exceeds a threshold.

- **Sigmoidal dependence model**:

    $$P_{release} = \frac{1}{1 + e^{-( [Ca^{2+}]_i - K_d ) / s}}$$

    where $K_d$ is the dissociation constant and $s$ is the slope parameter.

The amount of released neurotransmitter $N_{trans}$ can be modeled as a random variable, e.g., using a binomial distribution:

$$P(N_{trans} = k) = \binom{N_{ves}}{k} P_{release}^k (1 - P_{release})^{N_{ves} - k}$$

where $N_{ves}$ is the number of readily releasable vesicles.

**Connection to the Postsynaptic Neuron**

Released neurotransmitter binds to postsynaptic membrane receptors, generating a postsynaptic current as described in the dendrite section. Thus, the axon model connects to the dendrite model via synaptic transmission.

**Factors Affecting Action Potential Conduction Velocity**

Conduction velocity depends on several factors reflected in mathematical models:

- **Axon diameter**: Larger diameter reduces axial resistance, increasing conduction velocity.  
- **Myelination**: Presence and thickness of the myelin sheath greatly increase velocity via saltatory conduction.  
- **Ion channel density**: Voltage-gated sodium channel density at Nodes of Ranvier affects action potential generation efficiency.  
- **Temperature**: Temperature affects ion channel kinetics.

**Conclusion**

Mathematical axon formalization includes modeling action potential generation based on ion channel dynamics, describing signal propagation along the axon accounting for myelination, and modeling synaptic transmission. The Hodgkin-Huxley model is foundational, though simpler models exist for specific tasks. Understanding the mathematical basis of axonal function is critical for studying neural networks and developing neuromorphic technologies.

**Limitations and Future Directions:**  
- **Heterogeneity**: Significant heterogeneity exists in axon properties across neuron types.  
- **Plasticity**: Axonal properties—including channel conductance and myelination—can change over time.  
- **Molecular details**: Models can be extended to include more detailed molecular mechanisms of synaptic transmission.

## Synapses: The Site of Intercellular Signal Transmission

Synapses are specialized structures that transmit information from one neuron to another or to effector cells (muscle or glandular cells). There are two main types of synapses: chemical and electrical.

**Internal Synaptic Structure:**  
- **Chemical synapses**: The most common type of synapse. Signal transmission occurs via chemical messengers—neurotransmitters.  
  - **Presynaptic membrane**: The membrane of the axon terminal that releases neurotransmitters. It contains voltage-gated calcium channels and mechanisms for synaptic vesicle exocytosis.  
  - **Synaptic cleft**: A narrow space (approximately 20–40 nm) between the pre- and postsynaptic membranes.  
  - **Postsynaptic membrane**: The membrane of the target cell (usually a dendrite or soma of another neuron), containing neurotransmitter receptors.  
  - **Neurotransmitters**: Chemical substances synthesized by the neuron and stored in synaptic vesicles. Examples include glutamate (the primary excitatory neurotransmitter), GABA (the primary inhibitory neurotransmitter), acetylcholine, dopamine, serotonin, norepinephrine, etc. Different neurotransmitters exert distinct effects on the postsynaptic cell.  
  - **Receptors**: Protein molecules on the postsynaptic membrane with specific affinity for particular neurotransmitters. Neurotransmitter binding induces a conformational change in the receptor, opening or closing ion channels and thereby altering the postsynaptic membrane potential (EPSP or IPSP). There are two main receptor types:  
    - **Ionotropic receptors**: Ligand-gated ion channels. Neurotransmitter binding directly opens an ion channel, producing a fast and transient effect.  
    - **Metabotropic receptors**: Coupled to G-proteins. Neurotransmitter binding activates a G-protein, which either directly modulates ion channels or triggers second messenger cascades, resulting in slower and more prolonged effects.  
  - **Neurotransmitter inactivation**: To prevent continuous stimulation of the postsynaptic cell, neurotransmitters must be rapidly removed from the synaptic cleft. This occurs via:  
    - **Reuptake**: Transporter proteins on the presynaptic membrane or glial cells actively take up the neurotransmitter back into the presynaptic terminal or glial cell.  
    - **Enzymatic degradation**: Enzymes in the synaptic cleft break down the neurotransmitter into inactive components.  
- **Electrical synapses**: Less common than chemical synapses. Characterized by direct electrical coupling between pre- and postsynaptic cells via gap junctions.  
  - **Gap junctions**: Composed of protein channels (connexons) that connect the cytoplasm of two adjacent cells, allowing ions and small molecules to move freely between them.  
  - **Rapid signal transmission**: Signal transmission occurs almost instantaneously, without the delay characteristic of chemical synapses.  
  - **Bidirectional transmission**: Signals can propagate in both directions.  
  - **Activity synchronization**: Electrical synapses play a key role in synchronizing the activity of neuronal groups.

Understanding the structure of the biological neuron and synaptic signal transmission mechanisms is fundamental to studying nervous system function, developing treatments for neurological and psychiatric disorders, and designing more advanced artificial neural networks. Ongoing research in this field continues to reveal the intricate details of these remarkable cells.

### **Synapses – Mathematical Model:**

### Chemical Synapses: Mathematical Formalization

Signal transmission at a chemical synapse is a complex cascade of events that can be divided into several stages, each amenable to mathematical modeling.

**Presynaptic processes: action potential arrival and calcium influx**

Action potential arrival at the presynaptic terminal causes membrane depolarization, opening voltage-gated calcium channels. Calcium current dynamics $I_{Ca}$ can be described analogously to the Hodgkin-Huxley model:

$$I_{Ca} = P_{Ca} g_{Ca} s^p (V_{pre} - E_{Ca})$$

where:  
- $P_{Ca}$ — maximum calcium channel permeability.  
- $g_{Ca}$ — single calcium channel conductance.  
- $s$ — calcium channel activation variable, governed by a first-order differential equation.  
- $p$ — Hill coefficient reflecting channel opening cooperativity.  
- $V_{pre}$ — presynaptic terminal potential.  
- $E_{Ca}$ — calcium ion equilibrium potential.

Presynaptic terminal calcium concentration $[Ca^{2+}]_{pre}$ dynamics are described by:

$$\frac{d[Ca^{2+}]_{pre}}{dt} = - \alpha I_{Ca} - \frac{[Ca^{2+}]_{pre} - [Ca^{2+}]_{rest}}{\tau_{Ca}}$$

where:  
- $\alpha$ — constant linking calcium current to concentration change.  
- $[Ca^{2+}]_{rest}$ — resting calcium concentration.  
- $\tau_{Ca}$ — effective calcium removal time constant (buffering, extrusion).

**Neurotransmitter release: probabilistic models**

Neurotransmitter release is a quantal and probabilistic process. Vesicle release probability $P_{release}$ depends on presynaptic terminal calcium concentration. A more accurate model than a simple sigmoid accounts for multiple calcium-binding sites required for vesicle fusion:

$$P_{release} = \frac{[Ca^{2+}]_{pre}^n}{K_d^n + [Ca^{2+}]_{pre}^n}$$

where $n$ is the number of calcium ions required to trigger release, and $K_d$ is the dissociation constant.

The number of released vesicles can be modeled as a random variable following a binomial distribution (as previously noted) or a Poisson distribution if release probability is low and the number of potential release sites is large.

The dynamics of the readily releasable vesicle pool can be described by a differential equation:

$$\frac{dN_{ves}}{dt} = -k_{release} P_{release} N_{ves} + k_{refill} (N_{max} - N_{ves})$$

where:  
- $N_{ves}$ — number of readily releasable vesicles.  
- $k_{release}$ — release rate constant.  
- $k_{refill}$ — vesicle pool replenishment rate constant.  
- $N_{max}$ — maximum vesicle pool capacity.

**Neurotransmitter diffusion and binding in the synaptic cleft**

Neurotransmitter concentration $[NT]$ dynamics in the synaptic cleft can be described by a reaction-diffusion equation:

$$\frac{\partial [NT](x,t)}{\partial t} = D \nabla^2 [NT](x,t) - k_{bind} [NT](x,t) (R_{max} - [R_{bound}](x,t)) + k_{unbind} [R_{bound}](x,t) - k_{decay} [NT](x,t)$$

where:  
- $D$ — neurotransmitter diffusion coefficient.  
- $\nabla^2$ — Laplacian operator.  
- $k_{bind}$ — neurotransmitter-receptor binding rate constant.  
- $R_{max}$ — total receptor concentration.  
- $[R_{bound}]$ — bound receptor concentration.  
- $k_{unbind}$ — neurotransmitter-receptor unbinding rate constant.  
- $k_{decay}$ — neurotransmitter degradation or removal rate constant.

In most cases, for modeling simplicity, uniform neurotransmitter distribution in the cleft is assumed, reducing the equation to an ordinary differential equation (as in the provided text).

**Postsynaptic response: ionotropic and metabotropic receptors**

**Ionotropic receptors**: Neurotransmitter binding to an ionotropic receptor rapidly opens an ion channel. The ionotropic receptor current $I_{iono}$ can be modeled as:

$$I_{iono} = g_{iono} P_{open} (V_{post} - E_{rev})$$

where:  
- $g_{iono}$ — maximum channel conductance.  
- $P_{open}$ — channel open probability, dependent on bound neurotransmitter concentration. For a single-site model: $P_{open} = \frac{[NT]}{K_d + [NT]}$. Multisite models may exhibit more complex dependencies.  
- $V_{post}$ — postsynaptic membrane potential.  
- $E_{rev}$ — channel reversal potential.

**Metabotropic receptors**: Neurotransmitter binding to a metabotropic receptor activates a G-protein, triggering intracellular signaling cascades. Modeling this process can be complex, involving equations for G-protein, second messenger, and kinase concentrations. In the simplest case, metabotropic receptor effects on ion channel conductance can be modeled as:

$$g_{metabotropic}(t) = g_{max} \cdot f([NT](t))$$

where $f([NT](t))$ is a function describing conductance dependence on neurotransmitter concentration (e.g., a sigmoidal function). More detailed models account for G-protein and second messenger activation kinetics.

**Neurotransmitter inactivation**

Reuptake and enzymatic degradation processes affect the $k_{decay}$ constant in the neurotransmitter concentration equation. Reuptake can be modeled using the Michaelis-Menten equation for transport rate:

$$v_{uptake} = V_{max} \frac{[NT]}{K_m + [NT]}$$

where $V_{max}$ is the maximum transport rate and $K_m$ is the Michaelis-Menten constant. Enzymatic degradation is typically modeled as a first-order process.

### Electrical Synapses: Mathematical Formalization

Signal transmission at electrical synapses occurs through gap junctions, providing direct electrical coupling between cells. The gap junction current $I_{gap}$ between cells $i$ and $j$ can be described as:

$$I_{gap, ij} = g_{gap, ij} (V_i - V_j)$$

where:  
- $g_{gap, ij}$ — gap junction conductance between cells $i$ and $j$.  
- $V_i$ and $V_j$ — membrane potentials of cells $i$ and $j$, respectively.

Gap junction conductance $g_{gap, ij}$ may be constant or depend on various factors such as pH, calcium concentration, and membrane potential. Modeling gap junction conductance dynamics may include equations describing connexon opening and closing.

### Factors Influencing Synaptic Strength and Plasticity

Synaptic strength—the magnitude of the postsynaptic response to a presynaptic signal—depends on numerous factors that can change during synaptic plasticity:

- **Presynaptic factors**: Neurotransmitter release probability, number of readily releasable vesicles, presynaptic terminal calcium concentration.  
- **Postsynaptic factors**: Number and type of receptors, receptor affinity for neurotransmitter, efficiency of intracellular signaling cascades.  
- **Synaptic cleft properties**: Neurotransmitter diffusion and removal rates.

Mathematical models of synaptic plasticity describe the dynamics of these parameters as a function of synaptic activity (e.g., presynaptic spike frequency). Examples include models of long-term potentiation (LTP) and long-term depression (LTD) based on changes in AMPA receptor number and efficacy.

### Conclusion

Mathematical synapse formalization requires accounting for numerous interacting processes—from action potential arrival at the presynaptic terminal to postsynaptic potential generation. The choice of specific models and level of detail depends on the research question and available computational resources. Understanding the mathematical foundations of synaptic transmission is key to studying neural network function, learning and memory mechanisms, and developing novel therapeutic approaches for neurological and psychiatric disorders.

**Limitations and Future Directions:**  
- **Complexity and heterogeneity**: Synapses exhibit significant structural and functional complexity and heterogeneity.  
- **Stochasticity**: Many synaptic-level processes are stochastic, requiring probabilistic models.  
- **Multiscale integration**: Linking molecular-level models (e.g., receptor kinetics) with cellular and network-level models remains challenging.

# **The Artificial Neuron and Its Components**

## Introduction

Artificial neural networks (ANNs) are a class of computational models inspired by the structure and operating principles of biological neural networks. Historically, interest in artificial intelligence led to the development of mathematical models capable of mimicking cognitive functions. ANNs, emerging from this research, are a powerful tool for solving a wide range of tasks—from pattern recognition to natural language processing. It is important to understand that ANNs are not exact replicas of biological systems but rather mathematical abstractions of them. The purpose of this section is to examine in detail the structure and operating principles of the artificial neuron—the fundamental building block of any artificial neural network. Understanding its components and mechanisms is necessary for further comparison with biological counterparts and for appreciating the capabilities and limitations of modern neural network technologies.

## The Artificial Neuron and Its Components

### Structure of the Artificial Neuron

In its basic form, an artificial neuron is a mathematical function that processes input signals and generates an output signal. To understand its operation, consider its main components and draw analogies with the biological neuron:

1. **Inputs**: An artificial neuron receives multiple input signals, mathematically represented as numerical values ($x_1, x_2, ..., x_n$). These inputs are analogous to the dendrites of a biological neuron, which receive signals from other neurons.

2. **Weights**: Each input signal is associated with a specific weight ($w_1, w_2, ..., w_n$). Weights are numerical coefficients that determine the strength or importance of the corresponding input signal. Analogous to synaptic strength in a biological neuron, a weight determines how strongly an input signal will influence the artificial neuron’s activation.

3. **Aggregation function**: The weighted input signals are combined using an aggregation function. The most common function is simple summation of weighted inputs. This is analogous to the integration of incoming signals in the soma of a biological neuron.

4. **Activation function**: The result of the aggregation function is passed to an activation function. This function performs a nonlinear transformation of the aggregated signal, determining whether the neuron will be "activated" and the intensity of its output signal. The activation function plays a crucial role in enabling ANNs to model complex nonlinear relationships. Numerous types of activation functions exist, each with its own properties.

5. **Output**: The output signal of the artificial neuron is the result of applying the activation function. This signal ($y$) is passed to the inputs of other neurons in the network, analogous to how a biological neuron’s axon transmits signals to other cells.

![Structure_of_artificial_neuron](https://neerc.ifmo.ru/wiki/images/a/a5/%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD_%D1%81%D1%85%D0%B5%D0%BC%D0%B0.png)

### Inputs and Weights: Modeling Synaptic Connections

The input signals of an artificial neuron ($x_i$) are numerical values received from other neurons or from the external environment. Weights ($w_i$) are the key parameters of the artificial neuron that determine the strength of the connection between the current neuron and the source of the input signal. A large absolute weight value indicates strong influence of the corresponding input on neuron activation, whereas a weight close to zero indicates weak influence. Negative weights can model inhibitory connections.

In addition to weights, the concept of **bias** ($b$) is often used. Bias is an additional parameter added to the weighted sum of inputs. Its role is to shift the activation function, allowing the neuron to activate even with zero input signals or, conversely, preventing activation with nonzero inputs. Bias can be viewed as analogous to the excitation threshold in a biological neuron.

Mathematically, the weighted sum of inputs including bias is expressed as:

$$\sum_{i=1}^{n} w_i x_i + b$$

where:  
- $x_i$ — value of the $i$-th input signal  
- $w_i$ — weight associated with the $i$-th input signal  
- $n$ — total number of input signals  
- $b$ — bias value

### Aggregation Function: Summing Incoming Signals

The most common aggregation function in artificial neurons is the **weighted sum**, as shown above. This function simply sums the products of each input signal and its corresponding weight, then adds the bias. The result of this operation is the neuron’s **net input** (or simply *net*).

$$net = \sum_{i=1}^{n} w_i x_i + b$$

Although the weighted sum is the most prevalent, other aggregation functions exist. For example, some specialized architectures use distance-based functions between input vectors and weights, but these are far less common in standard models.

### Activation Function: Introducing Nonlinearity

The activation function plays a critically important role in the operation of an artificial neuron by introducing **nonlinearity** into the model. Without nonlinearity, a multilayer neural network would be equivalent to a single-layer perceptron, since sequential linear transformations can be reduced to a single linear transformation. Nonlinearity enables neural networks to approximate complex, nonlinear dependencies in data.

Consider the most common types of activation functions:

- **Threshold function**: The simplest activation function, which outputs 1 if the input exceeds a certain threshold and 0 otherwise. It is analogous to the "all-or-nothing" principle of action potential generation in biological neurons. Mathematically:

    $$f(x) = \begin{cases} 1, & \text{if } x \ge \theta \\ 0, & \text{if } x < \theta \end{cases}$$

    where $\theta$ is the threshold value.

- **Sigmoid function**: Also known as the logistic function, the sigmoid is a smooth, differentiable function that maps input values to the range 0 to 1. Historically very popular, its output can be interpreted as a probability. Mathematical form:

    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **Hyperbolic tangent (Tanh)**: The tanh function is similar to the sigmoid but has an output range from –1 to 1. This can be useful in some architectures where data centering is beneficial. Mathematical form:

    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **ReLU (Rectified Linear Unit)**: ReLU is one of the most popular activation functions in modern deep neural networks due to its simplicity and training efficiency. It returns the input value if positive and 0 otherwise. Mathematical form:

    $$f(x) = \max(0, x)$$

    Various ReLU variants exist, such as **Leaky ReLU** and **ELU (Exponential Linear Unit)**, which introduce a small slope for negative inputs to avoid the "dead neuron" problem, where a neuron stops activating.

    - **Leaky ReLU**: $f(x) = \begin{cases} x, & \text{if } x > 0 \\ \alpha x, & \text{if } x \le 0 \end{cases}$, where $\alpha$ is a small constant (e.g., 0.01).  
    - **ELU**: $f(x) = \begin{cases} x, & \text{if } x > 0 \\ \alpha (e^x - 1), & \text{if } x \le 0 \end{cases}$, where $\alpha$ is a positive constant.

- **Other activation functions**: Numerous other activation functions have been developed for specific tasks, such as **Softmax**, used in output layers for multiclass classification, and **Swish**, which has shown good performance in some architectures.

The choice of activation function is an important aspect of neural network design and depends on the specific task and network architecture. Different activation functions have different properties, such as differentiability, output range, computational complexity, and ability to mitigate vanishing or exploding gradients during training.

### Output of the Artificial Neuron

The output signal of the artificial neuron ($y$) is the result of applying the activation function ($\phi$) to the aggregated signal (*net*). Mathematically:

$$y = \phi(net) = \phi(\sum_{i=1}^{n} w_i x_i + b)$$

This output signal is then passed to the inputs of other neurons in subsequent network layers. Depending on the task, the output signal may represent a classification, regression value, or other data types. In multilayer neural networks, one neuron’s output becomes the input for neurons in the next layer, forming a complex web of interconnections.

## Mathematical Model of the Artificial Neuron

The entire operation of an artificial neuron can be summarized in a single mathematical equation describing the transformation of input signals into an output signal:

$$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$

where:  
- $y$ — neuron output signal  
- $\phi$ — activation function  
- $w_i$ — input signal weights  
- $x_i$ — input signals  
- $b$ — bias  
- $n$ — number of input signals

During neural network training, it is precisely the model parameters—**weights ($w_i$) and bias ($b$)**—that are adjusted so the network can perform the assigned task, such as correctly classifying images or predicting values. The activation function ($\phi$) is typically chosen in advance and remains unchanged during training.

## Conclusion

The artificial neuron, despite its apparent simplicity, is a powerful building block for constructing complex artificial neural networks. It is a mathematical model that mimics certain aspects of biological neuron operation, such as receiving input signals, weighting them, aggregating them, and generating an output signal. It is important to understand that the artificial neuron is a **simplified model** of the biological neuron, ignoring many complex biochemical and physiological processes occurring in real neural cells. Nevertheless, this simplified model has proven extremely effective for solving a wide range of tasks.

In the next part of our work, we will conduct a detailed comparison of the structure and function of biological and artificial neurons, identifying both similarities and fundamental differences between these systems.

# **Comparative Analysis of Biological and Artificial Neurons**

We now conduct an in-depth comparative analysis of biological and artificial neurons, examining their structural, functional, and computational aspects.

1. **Structural Comparison:**

**Biological neuron:**  
- **Soma**: A complex cellular structure containing a nucleus with genetic material and diverse organelles supporting metabolic processes and molecular synthesis. The cytoskeleton provides structural support and participates in intracellular transport.  
- **Dendrites**: Branched processes specialized for receiving incoming signals from other neurons via synapses. Dendritic morphology (number, length, branching) significantly affects signal integration. Dendritic spines increase surface area for synaptic contacts and exhibit plasticity.  
- **Axon**: A single long process designed to transmit outgoing signals as action potentials. The axon initial segment (axon hillock) plays a key role in action potential initiation. The axon may be myelinated, enabling fast saltatory conduction.  
- **Synapses**: Specialized structures enabling signal transmission between neurons. Chemical synapses use neurotransmitters released by the presynaptic neuron and binding to receptors on the postsynaptic neuron, causing excitatory or inhibitory postsynaptic potentials. Electrical synapses (gap junctions) provide direct, rapid ionic coupling between neurons.

**Artificial neuron:**  
- **Inputs**: Numerical values corresponding to the activity of previous neurons or input data. Input count determines input vector dimensionality.  
- **Weights**: Numerical parameters modeling synaptic connection strength. Positive weights correspond to excitatory connections, negative to inhibitory. Weights are tuned during training to optimize network performance.  
- **Aggregation function**: Typically a weighted sum of input signals. May include bias, analogous to neuronal excitation threshold.  
- **Activation function**: A nonlinear function applied to the aggregation result to determine neuron output. Different activation functions (sigmoid, ReLU, tanh, etc.) introduce nonlinearity, enabling complex dependency modeling.  
- **Output**: A numerical value representing neuron activity and passed to other neurons’ inputs.

2. **Functional Comparison:**

**Biological neuron:**  
- **Electrochemical signaling**: Information transmission is based on membrane potential changes caused by ion flow through ion channels.  
- **Complex ion channel systems**: Various voltage-gated and ligand-gated ion channels enable action potential generation and propagation and modulate synaptic transmission.  
- **Nonlinear signal transmission**: Nonlinearity arises at multiple levels, including ion channel behavior, synaptic transmission saturation, and dendritic integration.  
- **Adaptive synaptic plasticity**: Synaptic connection strength can change over time based on neuronal activity (synaptic plasticity, e.g., long-term potentiation and depression), forming the basis of learning and memory. Multiple forms of plasticity exist, dependent on timing and mechanisms.  
- **Continuous real-time operation**: Biological neurons operate asynchronously and continuously, processing information in real time.

**Artificial neuron:**  
- **Numerical values**: Information processing involves manipulation of numerical values.  
- **Weighting and summation mathematical operations**: The core mechanism for processing input signals.  
- **Nonlinear activation functions**: Introduce necessary nonlinearity for modeling complex functions. Activation function choice affects trainability and performance.  
- **Weight adjustment during training**: Learning occurs through iterative weight tuning using optimization algorithms (e.g., gradient descent) based on training data.  
- **Discrete data processing**: Most implementations process data discretely, although models simulating continuous time exist.

**Mathematical Models:**

**Biological neuron:**  
- **Hodgkin-Huxley model for action potential**:  
  $$C_m \frac{dV_m}{dt} = -I_{ion} + I_{ext}$$  
  where $C_m$ is membrane capacitance, $V_m$ is membrane potential, $t$ is time, $I_{ion}$ is total ionic current (including sodium, potassium, and leak currents), and $I_{ext}$ is externally applied current. This model describes membrane potential dynamics based on individual ion channel behavior and is fundamental to understanding action potential generation. Other, simpler models exist, such as the FitzHugh-Nagumo model, which preserves core dynamic properties with lower computational complexity.

**Artificial neuron:**  
- **Simple mathematical model**:  
  $$y = \phi(\sum_{i=1}^{n} w_i x_i + b)$$  
  where $y$ is neuron output, $\phi$ is activation function, $w_i$ is the weight of the $i$-th input, $x_i$ is the value of the $i$-th input, $b$ is bias. This model represents a significant simplification of the biological neuron, focusing on core principles of weighted summation and nonlinear transformation. More complex artificial neuron models exist, such as recurrent neurons (with feedback connections for sequential data processing) or convolutional neurons (specialized for spatially structured data).

**Key Conclusions from Comparative Analysis:**

1. **Core similarities:**  
- **Functional analogy in signal processing**: Both neuron types perform the fundamental function of receiving, processing, and transmitting information. Biological neurons integrate electrochemical signals; artificial neurons integrate numerical values.  
- **Input signal integration principle**: Both biological and artificial neurons sum incoming signals, though integration mechanisms differ significantly (spatiotemporal integration in dendrites vs. weighted summation).  
- **Presence of nonlinear transformation**: Nonlinearity is a key property of both neuron types, enabling complex dependency modeling. In biological neurons, nonlinearity arises from ion channel properties and synaptic transmission; in artificial neurons, from activation functions.  
- **Capacity for learning and adaptation**: Both biological and artificial neural networks can modify their parameters (synaptic strength or weights) in response to experience, enabling adaptation to new tasks and data. However, learning mechanisms differ fundamentally.

2. **Fundamental differences:**  
- **Information processing substrate**: The biological neuron relies on complex biochemical and electrophysiological processes involving ion movement, neurotransmitters, and membrane potentials. The artificial neuron operates with abstract mathematical operations on numbers.  
- **Structural and dynamic complexity**: Biological neurons exhibit far greater structural complexity at molecular, cellular, and network levels. Their dynamics involve numerous interacting processes not yet fully replicated in artificial models.  
- **Timescales and synchrony**: Biological neurons operate on millisecond timescales and can exhibit complex synchronous activity patterns. Artificial neural networks often operate in discrete time, though spiking neural network research aims for more realistic temporal dynamics modeling.  
- **Learning and adaptation mechanisms**: Biological learning involves complex biochemical processes like synaptic plasticity modulated by various neurotransmitters and factors. Artificial neural network learning typically uses optimization algorithms like backpropagation.  
- **Energy consumption**: Biological neurons demonstrate remarkable energy efficiency compared to modern computing systems implementing artificial neural networks.

3. **Advantages and Limitations:**

**Biological neuron:**  
+ **High adaptability and learnability**: Biological neural networks possess exceptional ability to adapt to new conditions and learn from limited examples.  
+ **Energy efficiency**: Operates with extremely low power consumption, a subject of active research in neuromorphic computing.  
+ **Parallel and distributed information processing**: Massive parallelism and distributed processing provide fault tolerance and high efficiency.  
– **Relatively slow signal transmission speed**: Action potential propagation speed is limited by physiological factors.  
– **Susceptibility to fatigue and biological constraints**: Neuron function depends on metabolic processes and can be disrupted by various factors.

**Artificial neuron:**  
+ **High computational speed**: Can perform mathematical operations significantly faster than biological neurons.  
+ **Ease of scaling and reproducibility**: ANN architecture and parameters can be easily scaled and reproduced.  
+ **Stability and predictability**: Properly configured ANNs demonstrate stable operation.  
– **Simplified information processing model**: Does not reflect the full complexity of biological neurons, limiting capabilities in some domains.  
– **High energy consumption**: Training and operating large ANNs require substantial computational resources and energy.  
– **Limited generalization and adaptation capacity**: May struggle to generalize to novel, unseen data and require large amounts of training examples.

This in-depth analysis demonstrates that the artificial neuron, despite its simplicity compared to its biological prototype, successfully mimics key information processing principles, enabling effective solutions to a wide range of AI tasks. However, it is crucial to understand the fundamental differences and limitations of artificial models and to continue research toward more biologically realistic neural networks. Future research directions include developing neuromorphic chips mimicking biological neural network architecture and dynamics, and deepening our understanding of biological learning and plasticity mechanisms for integration into artificial models.

### References

[1] Nature Neuroscience: The Hodgkin-Huxley theory of the action potential

[2] Keener, J., & Sneyd, J. (2008). Mathematical Physiology: I: Cellular Physiology. Springer.