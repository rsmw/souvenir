# Souvenir: A narrative scripting language

Souvenir is a scripting language designed for games with complex narrative content. It is largely a response to visual novel and adventure game DSLs (e.g. RenPy, Ink, KAG) with the addition of concurrency features inspired by Erlang and embedded hardware.

```souvenir
-- Print the lyrics of 99 Bottles of Beer

$count = 99

:: verse
> {$count} bottles of beer on the wall
> {$count} bottles of beer
> Take one down
> Pass it around

$count -= 1
if $count ?= 0
    > No more bottles of beer on the wall
    bye
else
    > {$count} bottles of beer on the wall
    >
;;

-> verse
```

The first implementation halted development in 2017. The interpreter has since been entirely rewritten to address performance limitations, remove dependencies, speed up compile times, and improve suitability for hosting inside high performance game engines. Support for Bevy integration is underway.
