Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

Inductive LeafCtorTree :=
  | LeafCtorTree_E.

Inductive LeafCtorColor :=
  | LeafCtorColor_R
  | LeafCtorColor_B.

Inductive CtorTree :=
  | CtorTree_E
  | CtorTree_T.

Inductive TupLeafCtorColorLeafCtorTreeLeafCtorTree :=
  | MkLeafCtorColorLeafCtorTreeLeafCtorTree : LeafCtorColor -> LeafCtorTree -> LeafCtorTree -> TupLeafCtorColorLeafCtorTreeLeafCtorTree.

Inductive TupLeafCtorColorCtorTreeCtorTree :=
  | MkLeafCtorColorCtorTreeCtorTree : LeafCtorColor -> CtorTree -> CtorTree -> TupLeafCtorColorCtorTreeCtorTree.

Definition genLeafColor (chosen_ctor : LeafCtorColor) (stack1 : nat) (stack2 : nat) : G (Color) :=
  match chosen_ctor with
  | LeafCtorColor_R => 
    (returnGen (R ))
  | LeafCtorColor_B => 
    (returnGen (B ))
  end.

Definition genLeafTree (chosen_ctor : LeafCtorTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match chosen_ctor with
  | LeafCtorTree_E => 
    (returnGen (E ))
  end.

Fixpoint genTree (size : nat) (chosen_ctor : CtorTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match size with
  | O  => match chosen_ctor with
    | CtorTree_E => 
      (returnGen (E ))
    | CtorTree_T => 
      (bindGen 
      (* Frequency2 *) (freq [
        (* 1 *) (match (stack1, stack2) with
        | (4, 4) => 63
        | (4, 6) => 48
        | (6, 4) => 71
        | (6, 6) => 53
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorTreeLeafCtorTree LeafCtorColor_R LeafCtorTree_E LeafCtorTree_E))); 
        (* 2 *) (match (stack1, stack2) with
        | (4, 4) => 51
        | (4, 6) => 66
        | (6, 4) => 36
        | (6, 6) => 62
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorTreeLeafCtorTree LeafCtorColor_B LeafCtorTree_E LeafCtorTree_E)))]) 
      (fun param_variantis => (let '(MkLeafCtorColorLeafCtorTreeLeafCtorTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 1) 
        (fun p1 => 
          (bindGen (genLeafTree ctor2 stack2 3) 
          (fun p2 => 
            (bindGen 
            (* GenZ1 *)
            (let _weight_1 := match (stack1, stack2) with
            | (4, 4) => 58
            | (4, 6) => 51
            | (6, 4) => 53
            | (6, 6) => 68
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (stack1, stack2) with
            | (4, 4) => 69
            | (4, 6) => 43
            | (6, 4) => 39
            | (6, 6) => 24
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (stack1, stack2) with
            | (4, 4) => 67
            | (4, 6) => 37
            | (6, 4) => 72
            | (6, 6) => 47
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
              (* GenZ3 *)
              (let _weight_1 := match (stack1, stack2) with
              | (4, 4) => 46
              | (4, 6) => 48
              | (6, 4) => 54
              | (6, 6) => 41
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (stack1, stack2) with
              | (4, 4) => 40
              | (4, 6) => 73
              | (6, 4) => 43
              | (6, 6) => 36
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (stack1, stack2) with
              | (4, 4) => 72
              | (4, 6) => 43
              | (6, 4) => 42
              | (6, 6) => 69
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
                (bindGen (genLeafTree ctor3 stack2 5) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorTree_E => 
      (returnGen (E ))
    | CtorTree_T => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 10
        | (1, 4, 6) => 10
        | (1, 6, 4) => 10
        | (1, 6, 6) => 10
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_R CtorTree_E CtorTree_E))); 
        (* 2 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 10
        | (1, 4, 6) => 12
        | (1, 6, 4) => 10
        | (1, 6, 6) => 10
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_B CtorTree_E CtorTree_E))); 
        (* 3 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 10
        | (1, 4, 6) => 12
        | (1, 6, 4) => 27
        | (1, 6, 6) => 10
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_R CtorTree_T CtorTree_E))); 
        (* 4 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 10
        | (1, 4, 6) => 14
        | (1, 6, 4) => 11
        | (1, 6, 6) => 11
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_B CtorTree_T CtorTree_E))); 
        (* 5 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 12
        | (1, 4, 6) => 10
        | (1, 6, 4) => 11
        | (1, 6, 6) => 13
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_R CtorTree_E CtorTree_T))); 
        (* 6 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 10
        | (1, 4, 6) => 13
        | (1, 6, 4) => 10
        | (1, 6, 6) => 10
        | (2, 0, 4) => 10
        | (2, 0, 6) => 10
        | (3, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_B CtorTree_E CtorTree_T))); 
        (* 7 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 90
        | (1, 4, 6) => 90
        | (1, 6, 4) => 90
        | (1, 6, 6) => 89
        | (2, 0, 4) => 90
        | (2, 0, 6) => 90
        | (3, 0, 0) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_R CtorTree_T CtorTree_T))); 
        (* 8 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 90
        | (1, 4, 6) => 89
        | (1, 6, 4) => 90
        | (1, 6, 6) => 90
        | (2, 0, 4) => 90
        | (2, 0, 6) => 90
        | (3, 0, 0) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorTreeCtorTree LeafCtorColor_B CtorTree_T CtorTree_T)))]) 
      (fun param_variantis => (let '(MkLeafCtorColorCtorTreeCtorTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 2) 
        (fun p1 => 
          (bindGen (genTree size1 ctor2 stack2 4) 
          (fun p2 => 
            (bindGen 
            (* GenZ2 *)
            (let _weight_1 := match (size, stack1, stack2) with
            | (1, 4, 4) => 40
            | (1, 4, 6) => 85
            | (1, 6, 4) => 56
            | (1, 6, 6) => 39
            | (2, 0, 4) => 37
            | (2, 0, 6) => 46
            | (3, 0, 0) => 52
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (size, stack1, stack2) with
            | (1, 4, 4) => 30
            | (1, 4, 6) => 48
            | (1, 6, 4) => 56
            | (1, 6, 6) => 78
            | (2, 0, 4) => 46
            | (2, 0, 6) => 66
            | (3, 0, 0) => 39
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (size, stack1, stack2) with
            | (1, 4, 4) => 75
            | (1, 4, 6) => 88
            | (1, 6, 4) => 50
            | (1, 6, 6) => 64
            | (2, 0, 4) => 23
            | (2, 0, 6) => 52
            | (3, 0, 0) => 34
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
              (* GenZ4 *)
              (let _weight_1 := match (size, stack1, stack2) with
              | (1, 4, 4) => 51
              | (1, 4, 6) => 54
              | (1, 6, 4) => 50
              | (1, 6, 6) => 27
              | (2, 0, 4) => 54
              | (2, 0, 6) => 49
              | (3, 0, 0) => 54
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (size, stack1, stack2) with
              | (1, 4, 4) => 49
              | (1, 4, 6) => 73
              | (1, 6, 4) => 59
              | (1, 6, 6) => 50
              | (2, 0, 4) => 45
              | (2, 0, 6) => 60
              | (3, 0, 0) => 65
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (size, stack1, stack2) with
              | (1, 4, 4) => 27
              | (1, 4, 6) => 84
              | (1, 6, 4) => 69
              | (1, 6, 6) => 37
              | (2, 0, 4) => 81
              | (2, 0, 6) => 88
              | (3, 0, 0) => 59
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
                (bindGen (genTree size1 ctor3 stack2 6) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 10
    end,
    (returnGen CtorTree_E)); 
    (* 2 *) (match (tt) with
    | tt => 90
    end,
    (returnGen CtorTree_T))]) 
  (fun init_ctor => (genTree 3 init_ctor 0 0))).

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
          
