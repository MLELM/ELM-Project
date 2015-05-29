<html>
<body>
<h1> Synopsis </h1>
We proposed a Game Theory based ELM algorithm for efficient regulation as well as adjusting ELM parameters. 

<h1> Objective </h1>
The main idea behind is separating the fixed variables apart from ones depending on regulators. Doing so avoids re-using the fixed variables, and improved the algorithm efficiency. Our proposed algorithm also uses the PCA-like analysis method, which discards the low-contribution components, and the running speed is further improved by 5 times.

<h1> File Organization </h1>
The respository contains 4 files:
<ul style=”list-style-type:disc”>  
<li> "compare.m" compares the performance of Game Theory based ELM with the original one in terms of running time and the testing accuracy. </li>
<li> "elm.m" implements the original ELM algorithm. </li>
<li> "elm_gt.m" implements our proposed Game Theory based ELM.</li>
<li> "elm_gt_predefine" computes the required information (H, H', Y and Y' in paper) to avoid re-using information.</li>
</ui>

<h1> Implementation </h1>
Compare performacne: Run "compare.m" in Matlab or Octave.
Use ELM API: Call "elm.m" in Matlab or Octave according to its API parameters.
Use API of Game Theory based ELM: Call "elm_gt.m" in Matlab or Octave according to its API parameters.

</html>  
