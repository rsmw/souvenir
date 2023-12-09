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

## Running scripts

The `Actor` type is the core Souvenir interpreter. Actors must run in a host environment that manages lifecycles, scheduling, and input/output. Use `GlobalHandle::with_values()` to create a shared global environment, and call `souvenir::parse()?.compile()?` to load a script. An Actor can then be spawned with `Actor::from_script()`.

A running Actor can be told to resume execution with `Actor::tick()`. Execution will proceed until the script performs a backwards jump (eg. in a loop) or generates an IO request. The host can call `Actor::poll_io()` to accept a pending IO request (of type `IoReq`), inspect it by calling `IoReq::payload()`, and respond by passing the request to `Actor::fulfill()`, which (if successful) will again resume execution. The status of an actor can be queried at any time with `Actor::status()` and `Actor::is_alive()`. If script evaluation results in an error, the actor will be killed, and the error can be viewed with `Actor::reap()`.

There is a simple example host implementation in `bin/tuihost.rs` that runs a single script in the terminal with limited IO capabilities.
