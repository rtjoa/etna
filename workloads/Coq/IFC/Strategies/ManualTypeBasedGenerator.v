From QuickChick Require Import QuickChick.

Require Import TestingCommon.
Require Import Reachability.
Require Import SSNI.
Require Import SanityChecks.
Require Import ZArith.
(* Require Import Generation. *)
From mathcomp Require Import ssreflect eqtype seq.
Import LabelEqType.


Derive Arbitrary for BinOpT.
Derive Arbitrary for Instr.
Derive Arbitrary for Pointer.
Derive Arbitrary for Value.
Derive Arbitrary for Atom.
Derive Arbitrary for Ptr_atom.
Derive Arbitrary for StackFrame.
Derive Arbitrary for Stack.
Derive Arbitrary for SState.
(* Derive Arbitrary for Variation. *)

Definition gen_variation :=
  fun s : nat =>
  unkeyed
    (fix arb_aux (size : nat) : G Variation :=
       match size with
       | 0 | _ =>
           bindGen arbitrary
             (fun p0 : Label =>
              bindGen arbitrary
                (fun p1 : SState =>
                 bindGen arbitrary (fun p2 : SState => returnGen (Var p0 p1 p2))))
       end) s
.

Definition VARIATION_SIZE := 2.

Definition test_propSSNI_smart :=
  forAll (gen_variation VARIATION_SIZE) (fun v =>
    propSSNI_smart default_table v
  ).

(*! QuickChick test_propSSNI_smart.  *)