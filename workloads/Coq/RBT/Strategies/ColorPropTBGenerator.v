Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

(* Look up in list of backtrack weights *)
Fixpoint get {a: Type} (l : list (nat * a)) (target_key : nat) (default : a): a :=
  match l with
  | [] =>
    (* This branch should never return *)
    default
  | (key, value) :: l' =>
    if Nat.eqb key target_key then
       value
    else get l' target_key default
  end.

Definition manual_gen_tree :=
    fun s : nat =>
    (let
       fix arb_aux (size : nat) : G Tree :=
         let weight_red := get [
          (1, 500); (2, 507); (3, 479); (4, 998); (5, 1)
         ] s 0 in
         let weight_leaf := 500 in
         match size with
         | 0 => returnGen E
         | S size' =>
             freq [ (weight_leaf, returnGen E);
             (1000 - weight_leaf,
             bindGen (freq [ (weight_red, returnGen R); (1000-weight_red, returnGen B)])
               (fun p0 : Color =>
                bindGen (arb_aux size')
                  (fun p1 : Tree =>
                   bindGen arbitrary
                     (fun p2 : Z =>
                      bindGen arbitrary
                        (fun p3 : Z =>
                         bindGen (arb_aux size')
                           (fun p4 : Tree => returnGen (T p0 p1 p2 p3 p4)))))))]
         end in
     arb_aux) s.

#[global]
Instance genTree : GenSized (Tree) := 
  {| arbitrarySized n := manual_gen_tree n |}.

(* --------------------- Tests --------------------- *)

Definition test_prop_InsertValid :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        (prop_InsertValid t k v)))).

(*! QuickChick test_prop_InsertValid. *)

Definition test_prop_DeleteValid :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
        prop_DeleteValid t k)).

(*! QuickChick test_prop_DeleteValid. *)

Definition test_prop_InsertPost :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
     forAll arbitrary (fun v =>
        prop_InsertPost t k k' v)))).

(*! QuickChick test_prop_InsertPost. *)

Definition test_prop_DeletePost := 
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeletePost t k k'))).

(*! QuickChick test_prop_DeletePost. *)
    
Definition test_prop_InsertModel :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        prop_InsertModel t k v))).

(*! QuickChick test_prop_InsertModel. *)
    
Definition test_prop_DeleteModel :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
            prop_DeleteModel t k)).

(*! QuickChick test_prop_DeleteModel. *)

Definition test_prop_InsertInsert :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
    forAll arbitrary (fun v' =>     
        prop_InsertInsert t k k' v v'))))).

(*! QuickChick test_prop_InsertInsert. *)
    
Definition test_prop_InsertDelete := 
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
        prop_InsertDelete t k k' v)))).

(*! QuickChick test_prop_InsertDelete. *)
    
Definition test_prop_DeleteInsert := 
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v' =>
        prop_DeleteInsert t k k' v')))).

(*! QuickChick test_prop_DeleteInsert. *)
    
Definition test_prop_DeleteDelete :=  
    forAll arbitrary (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeleteDelete t k k'))).

(*! QuickChick test_prop_DeleteDelete. *)


