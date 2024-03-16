---
layout: "article.njk"
title: Solving Linear Systems with Q-less Decomposition
subtitle : "How to solve Ax=b efficiently"
tags: [julia, linear algebra]
author: Robert Gregg
date: 2024-03-02
image: "/assets/images/equation.svg"
imageAlt: "Test"
---

I ran into an interesting problem when studying algorithms that perform causal discovery. As these algorithms progress they need to perform linear regressions on many subsets of the data. One might regress $x_1$ and $x_2$ onto variable $x_3$, calculate a score, then add an additional predictor and recalculate that score. Performing linear regressions amounts to solving a system of equations:

$$
x = A\setminus b
$$

where each column in $A$ is a predictor, $b$ is outcome, and $x$ is a vector of coefficients for the regression. There's some nuance to this. To perform a regression with an intercept term you need to append a column of 1's to the $A$ matrix. Additionally, a number of algorithms are available to perform this matrix inverse, but many conventional softwares default to a method called $QR$ decomposition. More interestingly, some software packages are clever and allow you to save this decomposition to be reused if $b$ changes.

Causal algorithms will change $b$ and $A$ so it would appear that saving the $QR$ decomposition would be pointless. However, $A$ doesn't change "too much". We only ever takes subsets of $A$ as the causal algorithm iterators over multiple predictors. If we compute the decomposition for the entire $A$ matrix, can we modify this decomposition slightly to solve a subsystem? If you guessed yes you're correct, otherwise this would be a pretty boring blog-post. :wink:

# Q-less QR Decomposition

$QR$ decomposition can be used to solve linear regression problems due to the properties of the decomposed matrices. $Q$ is orthogonal meaning it's inverse and transpose are equal, and $R$ is an upper triangular matrix so [forward/backward substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution) can be utilized. If $A$ is a square matrix, then the solution to $Ax=b$ becomes:

$$
x = R^{-1} Q^\intercal b
$$

You can see once we have the decomposition, then substituting different outcome variables (i.e., different $b$ vectors) becomes trivial. However, as mentioned before, $Q$ and $R$ change whenever $A$ changes, so how can we use the above equation? The answer comes in two parts: eliminating $Q$ from the equation and Givens Rotations. 

To eliminate $Q$ from the equation we can make use of the [Normal Equation](https://mathworld.wolfram.com/NormalEquation.html). Through some clever substitutions we can arrive at a solution for $x$ that does not depend on $Q$.

$$
\begin{align*}
Ax &= b && \text{Original equation}\\
A^\intercal Ax &= A^\intercal b && \text{Multiply by }A^\intercal \\
(QR)^\intercal QRx &= A^\intercal b && \text{Substitute } A=QR \\
R^\intercal Q^\intercal QRx &= A^\intercal b && \text{Rule of transposes } (QR)^\intercal = R^\intercal Q^\intercal \\
R^\intercal R x &= A^\intercal b && \text{Q is orthogonal by definition, } Q^\intercal Q=I \\
R^{-1}R^{-\intercal}A^\intercal b &= x && \text{Solving for x}
\end{align*}
$$

This can be solved efficiently in Julia using "in-place" functions that will not allocate any additional memory. Provided the variables A, its QR decomposition, b, and x are already defined:

```julia
using LinearAlgebra

mul!(x, A', b)
ldiv!(R', x)
ldiv!(R, x)
```

Note that we effectively have to solve two systems of equations, one for $R^\intercal$ and one for $R$. Fortunately, both triangular matrices and can be solved in $\mathcal{O}(n^2)$ time. 

# Givens Rotations

When you take a subset of columns from $A$, you have to then remove those same columns from $R$. This will more than likely cause $R$ to no longer be an upper triangular matrix, defeating the purpose of saving it in the first place. Our saving grace here is a technique called [Givens Rotations](https://en.wikipedia.org/wiki/Givens_rotation). These rotations are represented by orthogonal  matrices that rotate vectors in a given plane counterclockwise specified by a desired angle. This additional matrices can be "absorbed" into $Q$ because they are orthogonal and they have the interesting property that they can zero out any desired value in a matrix. 

This might be best demonstrated with an example. Take the following matrix $A$ and its QR decomposition:

$$
A = \begin{bmatrix}
0 & 0 & 1\\
0 & 1 & 0\\
1 & 1 & 1
\end{bmatrix}
$$

$$
Q = \begin{bmatrix}
0 & 0 & -1\\
0 & 1 & 0\\
-1 & 0 & 0
\end{bmatrix}
$$

$$
R = \begin{bmatrix}
-1 & -1 & -1\\
0 & 1 & 0\\
0 & 0 & -1
\end{bmatrix}
$$

You can check that these matrices satisfy all the required properties. $Q$ should be an orthogonal matrix and $R$ should be an upper triangular matrix. Now say we wanted to solve a linear system but remove the second column in $A$. Using Q-less decomposition, we know only $R$ needs to be modified to remain an upper triangular matrix. Here $R_{sub}$ only contains the 1st and 3rd columns of $R$.

$$
R_{sub} = \begin{bmatrix}
-1 & -1\\
0 & 0\\
0 & -1
\end{bmatrix}
$$

$R_{sub}$ needs to triangular, however, this is not the case after removing the second column. To "zero-out" the last entry in $R_{sub}$ we can apply the following Givens rotations:

$$
G = \begin{bmatrix}
1 & 0 & 0\\
0 & 0 & -1\\
0 & 1 & 0
\end{bmatrix}
$$

$$
G\cdot R_{sub} = \begin{bmatrix}
-1 & -1\\
0 & 1\\
0 & 0
\end{bmatrix}
$$

The product $G*Rsub$ reestablishes the upper-triangular requirement and, because $G$ is orthogonal, the $QR$ decomposition remains valid. What's great about this approach is that $G$ never needs to be explicitly formed as we are just modifying individual entries in $R_{sub}$. What's more, we don't have to worry about what happens to $Q$ because we have a solution that that only depends on $R$.

# Putting it all together

The following Julia code show how one might use this in practice.

```julia
#Loop through lower triangular matrix
#the 1st axis is reversed because we're moving non-zeros *up* above diagonal
lowTriIter(A::AbstractMatrix) = ((i,j) for i in reverse(axes(A,1)), j in axes(A,2) if i>j)

function qless!(x,A,b,R,Rsub,columnsKeep)

    numCol = length(columnsKeep)

    #Selecting the columns to keep
    for (k,j) in enumerate(columnsKeep)
        @views Rsub[:,k] = R[:,j]
    end

    #Use a givens rotation to zero out values below diagonal
    for (i,j) in lowTriIter(Rsub)        
        if Rsub[i,j] ≠ 0.0
            G,r = givens(Rsub,i-1,i,j)
            Rsub[i-1,j] = r
            Rsub[i,j] = 0.0

            #Givens impact columns to the right
            for k in j+1:numCol
                (r1, r2) = Rsub[i-1,k], Rsub[i,k]
                Rsub[i-1,k] = G.c*r1 + G.s*r2
                Rsub[i,k] = -G.s*r1 + G.c*r2
            end
        end
    end

    Rv = UpperTriangular(view(Rsub,1:numCol,1:numCol))

    #Solve R'R*b = X'y
    mul!(x,A',b)
    ldiv!(Rv',x)
    ldiv!(Rv,x)

    return nothing
end
```

# Potential Numerical Instability

A way to measure how accurate a solution $x$ is for the given problem uses a property called the condition number:

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|  
$$

Provided the error in the solution to $Ax=b$ is small, the condition number for the linear regression will also remain small. They should scale linearly with each other. However, because we are using the "normal equation" form to solve our system, we effectively square our condition number:

$$
\kappa(A^\intercal A) = \kappa(A)^2  
$$

This amplifies errors that occur due to machine precision. Where we might get 9 digits of accuracy before, now we only get 3. This is, of course, dependent on the matrix A, so  it's condition number should be checked before using Q-less decomposition. 
