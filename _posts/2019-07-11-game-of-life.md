---
layout: post
title:  "How to Write the Game of Life in Bash"
categories: bash game-of-life
---

The rules for the voting variation of the game of life are as follows:

Each cell in the grid (referred to as "world" in my program) is either 1 or 0. It's next state is determined by its neighbors and itself. If the majority are 1, then so is the next state. Same with 0. This variation is therefore smoothing, and eventually the world will stop moving. This game of life might not be interesting to look at, but it's simple to implement; I do so whenever I'm teaching myself a new language and my basic model becomes more refined every time.

<figure>
  <img src="{{site.url}}/assets/next-state.jpg" />
  <figcaption>Most neighbors are 0, so the cell's next state is 0.</figcaption>
</figure>

Our world will wrap somewhat like a globe. That is, the neighbors of the cells on the periphery will be on the opposite edge of the grid.

First, our script will check for arguments. `-eq` is a bash built-in that returns true if both operands are equal. Note that spacing is hugely important in bash. Assignment, for example, won't work unless there's no spacing between the operands.

```
if [ $# -eq 2 ]; then
    height=$1;
    width=$2;
else
    echo "Input integers to specify height and width. Using defaults."
    height=10;
    width=10;
fi
```

The above sets the height and width to defaults if they're not specified. The variables are always preceded by `$` when their values are accessed.

A hash is going to be the data structure that represents our grid, which we create like so

```
declare -A world
```

Now to populate the world with random 1's and 0's.

```
for i in `seq $height`; do
    for j in `seq $width`; do
        world[$i,$j]=$(( RANDOM % 2))
    done
done
```

The backticks (\`) execute the command within them. According to [tldp.org](http://tldp.org/LDP/abs/html/randomvar.html), `RANDOM` is `an internal Bash function (not a constant) that returns a pseudorandom integer in the range 0 - 32767.`

Then the following will loop every time the user hits `<ENTER>` in the terminal.

```
while read -e; do
    # print the world
    for i in `seq $height`; do
        for j in `seq $width`; do
            printf "%2s" ${world[$i,$j]}
        done
        echo ""
    done
    # set the world
    for i in `seq $height`; do
        for j in `seq $width`; do
            if [ `neighbor_sum` -gt 2 ]; then
                world[$i,$j]=1;
            else
                world[$i,$j]=0;
            fi
        done
    done
done
```

The first pair of for-loops simply prints the world. It's essential that we enclose the accessing of the hash in curly braces like `${world[$i,$j]}` in order to access the value associated with the key.

The second pair of for-loops will create the next state of the world. `neighbor_sum` is a function we're going to create next. Note that we're using `-gt` to test whether the result of neighbor_sum is greater than 2, not 4 as you would if we were considering diagonal cells as neighbors. If we did, that would give us 9 neighbors. We're not going to do that in this post for the sake of keeping the code simple, but this implementation could easily be advanced to that.

In the following, i and j are the same i and j of the loop in which they are called.

```
neighbor_sum() {
    local left_i=`left_or_above $i $width`
    local right_i=`right_or_below $i $width`
    local below_j=`right_or_below $j $height`
    local above_j=`left_or_above $j $height`
    echo `expr ${world[$left_i,$j]} + ${world[$right_i,$j]} + ${world[$i,$below_j]} + ${world[$i,$above_j]} + ${world[$i,$j]}`
}
```

Whenever we do arithmetic, we execute `expr` in a subshell. We also execute our functions in subshells and capture whatever they `echo` out. This is not ideal because it adds a lot overhead. In a future post, I may profile this code and see if I can shave off some time by doing the arithmetic with bash built-ins and by passing back the return value of each function by storing it in a variable.

The function `left_or_above` is so named because, depending on what parameters are passed, the function computes the i index of the cell to the left or the j index of the cell to the right. The two situations are conceptually analogous if you think about leftward and above as going towards 1. Likewise, rightward and below are advancing towards the maximum indices, width and height respectively.

```
right_or_below () {
    if [ $1 -eq $2 ]; then
        echo 1
    else
        echo `expr $1 + 1`
    fi
}

left_or_above () {
    if [ $1 -eq 1 ]; then
        echo $2
    else
        echo `expr $1 - 1`
    fi
}
```

And we're done. If we wanted, it would be a simple matter to bump up the number of neighbors considered from 5 to the full 9. After all, the cell above and to the left is given by the i value of the neighbor to the left and the j value of the neighbor above.

<figure>
  <img src="{{site.url}}/assets/above-left-neighbor.jpg" />
  <figcaption>Neighbor above and to the left.</figcaption>
</figure>
