Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

Definition genColor (size : nat) : G (Color) :=

  (* Frequency1 *) (freq [
    (* R *) (match (size) with
    | (0) => 49
    | _ => 500
    end,
    (returnGen (R ))); 
    (* B *) (match (size) with
    | (0) => 51
    | _ => 500
    end,
    (returnGen (B )))]).

Fixpoint genTree (size : nat) : G (Tree) :=
  match size with
  | O  => 
    (* Frequency2 (single-branch) *) 
    (returnGen (E ))
  | S size1 => 
    (* Frequency3 *) (freq [
      (* E *) (match (size) with
      | (1) => 50
      | (2) => 58
      | (3) => 85
      | (4) => 2
      | _ => 500
      end,
      (returnGen (E ))); 
      (* T *) (match (size) with
      | (1) => 50
      | (2) => 40
      | (3) => 0
      | (4) => 82
      | _ => 500
      end,
      (bindGen (genColor 0) 
      (fun p1 => 
        (bindGen (genTree size1) 
        (fun p2 => 
          (bindGen 
          (* GenZ1 *)
          (let _weight_1 := match (size) with
          | (1) => 50
          | (2) => 50
          | (3) => 50
          | (4) => 49
          | _ => 500
          end
          in
          bindGen (freq [
            (_weight_1, returnGen 1%Z);
            (100-_weight_1, returnGen 0%Z)
          ]) (fun n1 =>
          (let _weight_2 := match (size) with
          | (1) => 50
          | (2) => 50
          | (3) => 50
          | (4) => 51
          | _ => 500
          end
          in
          bindGen (freq [
            (_weight_2, returnGen 2%Z);
            (100-_weight_2, returnGen 0%Z)
          ]) (fun n2 =>
          (let _weight_4 := match (size) with
          | (1) => 50
          | (2) => 50
          | (3) => 50
          | (4) => 50
          | _ => 500
          end
          in
          bindGen (freq [
            (_weight_4, returnGen 4%Z);
            (100-_weight_4, returnGen 0%Z)
          ]) (fun n4 =>
            returnGen (n1 + n2 + n4)%Z
          )))))) 
          (fun p3 => 
            (bindGen 
            (* GenZ2 *)
            (let _weight_1 := match (size) with
            | (1) => 50
            | (2) => 50
            | (3) => 50
            | (4) => 49
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (size) with
            | (1) => 50
            | (2) => 50
            | (3) => 50
            | (4) => 51
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (size) with
            | (1) => 50
            | (2) => 50
            | (3) => 51
            | (4) => 49
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_4, returnGen 4%Z);
              (100-_weight_4, returnGen 0%Z)
            ]) (fun n4 =>
              returnGen (n1 + n2 + n4)%Z
            )))))) 
            (fun p4 => 
              (bindGen (genTree size1) 
              (fun p5 => 
                (returnGen (T p1 p2 p3 p4 p5)))))))))))))])
  end.

Definition gSized :=
  (genTree 4).

(* --------------------- Tests --------------------- *)

Definition test_prop_InsertValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        (prop_InsertValid t k v)))).

(*! QuickChick test_prop_InsertValid. *)

Definition test_prop_DeleteValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
        prop_DeleteValid t k)).

(*! QuickChick test_prop_DeleteValid. *)

Definition test_prop_InsertPost :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
     forAll arbitrary (fun v =>
        prop_InsertPost t k k' v)))).

(*! QuickChick test_prop_InsertPost. *)

Definition test_prop_DeletePost := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeletePost t k k'))).

(*! QuickChick test_prop_DeletePost. *)
    
Definition test_prop_InsertModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        prop_InsertModel t k v))).

(*! QuickChick test_prop_InsertModel. *)
    
Definition test_prop_DeleteModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
            prop_DeleteModel t k)).

(*! QuickChick test_prop_DeleteModel. *)

Definition test_prop_InsertInsert :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
    forAll arbitrary (fun v' =>     
        prop_InsertInsert t k k' v v'))))).

(*! QuickChick test_prop_InsertInsert. *)
    
Definition test_prop_InsertDelete := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
        prop_InsertDelete t k k' v)))).

(*! QuickChick test_prop_InsertDelete. *)
    
Definition test_prop_DeleteInsert := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v' =>
        prop_DeleteInsert t k k' v')))).

(*! QuickChick test_prop_DeleteInsert. *)
    
Definition test_prop_DeleteDelete :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeleteDelete t k k'))).

(*! QuickChick test_prop_DeleteDelete. *)
          
